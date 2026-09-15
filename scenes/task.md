# Bench2Dex release task catalog


| ID | Task | Scene |
|---:|---|---|
| 03 | Wine Glass Plate Balance | `03_wine_glass_plate_balance.yaml` |
| 06 | Fruit Bowl Loading | `06_fruit_bowl_loading.yaml` |
| 07 | Citrus Plate Loading | `07_citrus_plate_loading.yaml` |
| 08 | Frypan Stand Pour | `08_frypan_stand_pour.yaml` |
| 09 | Cleaner Moisturizer Box Loading | `09_cleaner_moisturizer_box_loading.yaml` |
| 12 | Screwdriver Box and Hammer | `12_screwdriver_box_and_hammer.yaml` |
| 21 | Condiment Box Loading | `21_condiment_box_loading.yaml` |
| 22 | Tool Box Loading | `22_tool_box_loading.yaml` |
| 24 | Stationery Category Sorting | `24_stationery_category_sorting.yaml` |
| 26 | Canned Food Tray Line Arrangement | `26_canned_food_tray_line_arrangement.yaml` |
| 27 | Ball Box Loading | `27_ball_box_loading.yaml` |
| 32 | Baking Tray Prep with Tools | `32_baking_tray_prep_with_tools.yaml` |
| 34 | Fridge Wine Interhand Pour | `34_fridge_wine_interhand_pour.yaml` |
| 42 | Trash Disposal | `42_trash_disposal.yaml` |
| 43 | Fridge Fruit Shelf Sorting | `43_fridge_fruit_shelf_sorting.yaml` |
| 44 | Microwave Bowl Loading | `44_microwave_bowl_loading.yaml` |
| 51 | Toilet Lid Cleaner Pour | `51_toilet_lid_cleaner_pour.yaml` |
| 60 | Breadbasket Fast Food Loading | `60_breadbasket_fast_food_loading.yaml` |
| 61 | Medicine Shoebox Pack | `61_medicine_shoebox_pack.yaml` |
| 62 | Shoebox Accessory Pack | `62_shoebox_accessory_pack.yaml` |
| 64 | Sports Ball Cup Sort | `64_sports_ball_cup_sort.yaml` |
| 67 | Faucet Cup Water Fill | `67_faucet_cup_water_fill.yaml` |
| 73 | Jigsaw Puzzle Assembly | `73_jigsaw_puzzle_assembly.yaml` |
| 76 | Soup Serving | `76_soup_serving.yaml` |
| 79 | Bimanual Piano Melody | `79_bimanual_piano_melody.yaml` |
| 80 | Gaming Desk Setup | `80_gaming_desk_setup.yaml` |

Run a task with, for example:

```bash
python main.py --task scenes/03_wine_glass_plate_balance.yaml
```

Task success criteria live in each YAML file. Custom evaluators referenced by those files live under `success/custom/`.
