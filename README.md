# pfib_capstone_project

```
cd pfib_puma/
```
To generate synthetic data according to config

```
python script/generate_synthetic_pfib.py --spec data/config/name_of_the_config.txt
```

To convert png slices into stl:
```
python scripts/convert_mesh.py \
  --png_dir output/puma_synthetic_sphere/png_slices \
  --out_mesh pfib_mesh.stl \
  --solid_is_bright \
  --step_size 2
```
