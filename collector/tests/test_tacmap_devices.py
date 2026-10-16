"""Exercise the sensor's tensor paths without launching Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


SOURCE = Path(__file__).resolve().parents[1] / "tacmap_sensor/sharpa_tacmap_vbts.py"


def sensor_method(name, namespace):
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[method], type_ignores=[])
    # Annotations refer to Isaac types; postpone their evaluation in this harness.
    import __future__
    exec(compile(module, str(SOURCE), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class TacMapDeviceTests(unittest.TestCase):
    def test_compute_device_is_set_before_mesh_creation_and_after_reset(self):
        class Base:
            meshes = {}

            def _initialize_warp_meshes(self):
                self.mesh_device = self._device

        tree = ast.parse(SOURCE.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        cls.bases = [ast.Name(id='Base', ctx=ast.Load())]
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_initialize_warp_meshes']
        tree.body = [cls]
        namespace = {'Base': Base, 'torch': torch, 'MultiMeshRayCaster': Base,
                     'wp': SimpleNamespace(get_device=lambda device: device)}
        exec(compile(ast.fix_missing_locations(tree), str(SOURCE), 'exec'), namespace)
        sensor = namespace['SharpaTacmap']()
        # Meta tensors test device migration without requiring a CUDA driver.
        sensor.cfg = SimpleNamespace(compute_device='meta')
        prim = SimpleNamespace(GetPath=lambda: '/target')
        sensor._raycast_targets_cfg = [SimpleNamespace(prim_expr='/target')]
        fake_isaaclab = SimpleNamespace(sim=SimpleNamespace(find_matching_prims=lambda _: [prim]))
        with mock.patch.dict('sys.modules', {'isaaclab': fake_isaaclab}):
            for _ in range(2):
                # SensorBase recreates these on the physics device after a reset.
                sensor._device = 'cpu'
                for name in ('_is_outdated', '_timestamp', '_timestamp_last_update'):
                    setattr(sensor, name, torch.ones(1))
                sensor._initialize_warp_meshes()
                self.assertEqual(sensor.mesh_device, 'meta')
                for name in ('_is_outdated', '_timestamp', '_timestamp_last_update'):
                    self.assertEqual(getattr(sensor, name).device.type, 'meta')
            Base.meshes['/target'] = SimpleNamespace(points=SimpleNamespace(device='cpu'))
            with self.assertRaisesRegex(RuntimeError, 'cached on cpu'):
                sensor._initialize_warp_meshes()

    def check_cpu_physics_poses(self, compute_device):
        device = torch.device(compute_device)
        math_utils = SimpleNamespace(
            combine_frame_transforms=lambda p, q, offset, rotation: (p + offset, q),
            quat_apply=lambda q, vectors: vectors.clone(),
            convert_quat=lambda q, to: q[:, [3, 0, 1, 2]],
        )
        attached_pose = [torch.tensor([[[1., 2., 3.]]]), torch.tensor([[[1., 0., 0., 0.]]])]
        # Both the attached body and target physics views return CPU tensors.
        rigid = object()
        link = SimpleNamespace(get_link_transforms=lambda: torch.tensor([[
            [0., 0., 0., 0., 0., 0., 1.], [7., 8., 9., 0., 0., 0., 1.]
        ]]))
        attached = object()

        def obtain(view, indices):
            self.assertIsNone(indices)
            if view is attached:
                return tuple(attached_pose)
            return torch.tensor([[[4., 5., 6.]]]), torch.tensor([[[1., 0., 0., 0.]]])

        calls = []

        def raycast(starts, directions, **kwargs):
            self.assertEqual(starts.device, device)
            self.assertEqual(kwargs['mesh_positions_w'].device, device)
            self.assertTrue(torch.isfinite(starts).all())
            calls.append(starts.clone())
            depths = torch.tensor([[0.01, float('inf')]] if len(calls) == 1 else [[0.02, 0.02]],
                                  device=device)
            return starts.clone(), depths, None, None, None

        namespace = {'torch': torch, 'math_utils': math_utils,
                     'obtain_world_pose_from_view': obtain,
                     'MultiMeshRayCaster': SimpleNamespace(mesh_offsets={}),
                     'raycast_dynamic_meshes': raycast}
        update_rays = sensor_method('_update_ray_infos', namespace)
        update = sensor_method('_update_buffers_impl', namespace)
        sensor = SimpleNamespace(
            _device=str(device), device=str(device), _num_envs=1, num_rays=2,
            _view=attached, _offset_pos=torch.zeros(1, 3, device=device),
            _offset_quat=torch.tensor([[1., 0., 0., 0.]], device=device),
            ray_starts_att=torch.zeros(1, 2, 3, device=device),
            ray_directions_att=torch.tensor([[[0., 0., 1.], [0., 0., 1.]]], device=device),
            _ray_starts_w=torch.zeros(1, 2, 3, device=device),
            _ray_directions_w=torch.zeros(1, 2, 3, device=device),
            _frame=torch.zeros(1, device=device),
            _mesh_views=[rigid, link],
            _raycast_targets_cfg=[SimpleNamespace(track_mesh_transforms=True, prim_expr=p)
                                  for p in ('rigid', 'link')],
            _articulation_link_indices={1: 1},
            _mesh_positions_w=torch.zeros(1, 2, 3, device=device),
            _mesh_orientations_w=torch.zeros(1, 2, 4, device=device),
            ray_hits_w=torch.zeros(1, 2, 3, device=device), _mesh_ids_wp=None,
            cfg=SimpleNamespace(max_distance=0.015, cpd_max_dist=0.5, update_mesh_ids=False),
            _data=SimpleNamespace(pos_w=torch.zeros(1, 3, device=device), output={
                'distance_along_normal_m': torch.zeros(1, 2, 1, device=device),
                'contact_mask': torch.zeros(1, 2, 1, dtype=torch.bool, device=device),
            }),
        )
        # view.count is needed for target layout; use a separate rigid view object.
        rigid_view = SimpleNamespace(count=1)
        sensor._mesh_views[0] = rigid_view
        link.count = 1
        sensor._update_ray_infos = lambda ids: update_rays(sensor, ids)
        ids = torch.tensor([0], device=device)
        update(sensor, ids)
        torch.testing.assert_close(sensor._ray_starts_w[0, 0],
                                   torch.tensor([1., 2., 3.], device=device))
        torch.testing.assert_close(sensor._mesh_positions_w[0],
                                   torch.tensor([[4., 5., 6.], [7., 8., 9.]], device=device))
        self.assertEqual(sensor._data.output['contact_mask'].flatten().tolist(), [True, False])
        torch.testing.assert_close(sensor._data.output['distance_along_normal_m'].flatten(),
                                   torch.tensor([0.01, 0.], device=device))
        # A new physics pose must be read on the next sample, not cached at reset.
        attached_pose[0] += 1.
        update_rays(sensor, ids)
        torch.testing.assert_close(sensor._ray_starts_w[0, 0],
                                   torch.tensor([2., 3., 4.], device=device))

    def test_live_cpu_pose_shapes_links_and_missed_rays(self):
        self.check_cpu_physics_poses('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'requires a visible CUDA device')
    def test_cpu_physics_to_cuda_tactile_buffers(self):
        self.check_cpu_physics_poses('cuda:0')


if __name__ == '__main__':
    unittest.main()
