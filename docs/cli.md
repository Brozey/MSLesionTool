# Command-line interface

The CLI runs the same ensemble as the GUI, headlessly.

```bash
python -m msseg.cli [options]
```

## Single patient

```bash
python -m msseg.cli --flair patient_flair.nii.gz --t1 patient_t1.nii.gz -o seg.nii.gz
```

## Batch

```bash
python -m msseg.cli --batch /data/subjects/ --output-dir /data/results/
```

FLAIR/T1 files are auto-assigned per subject from the input folder.

## ONNX backend

```bash
python -m msseg.cli --flair f.nii.gz --t1 t.nii.gz -o seg.nii.gz --backend onnx
```

Requires exported ONNX models (`python download_models.py --onnx` or `scripts/export_onnx.py`).

## Notes

- Inputs must be co-registered FLAIR + T1 in NIfTI format.
- Model weights download automatically on first use, or fetch them with `python download_models.py`.
- Use `--help` for the full option list.
