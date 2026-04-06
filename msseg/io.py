"""
msseg.io – NIfTI loading, file discovery, and result saving.
"""
import os
import numpy as np

# Optional imports
try:
    import SimpleITK as sitk
    _HAS_SITK = True
except ImportError:
    sitk = None
    _HAS_SITK = False

try:
    import nibabel as nib
    _HAS_NIB = True
except ImportError:
    nib = None
    _HAS_NIB = False


def load_nifti(path):
    """Load a NIfTI file, return (3D numpy array, affine, spacing, sitk_image)."""
    if _HAS_SITK:
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)  # shape: (D, H, W)
        spacing = img.GetSpacing()  # (sx, sy, sz)
        return arr, np.eye(4), spacing, img
    elif _HAS_NIB:
        nii = nib.load(path)
        arr = np.asanyarray(nii.dataobj)
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 1, 0))  # to (D, H, W)
        affine = nii.affine
        spacing = tuple(np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)))
        return arr, affine, spacing, nii
    else:
        raise RuntimeError("SimpleITK or nibabel required to load NIfTI files.")


def find_nifti_files_recursive(folder):
    """Recursively find all NIfTI files in a folder and subfolders."""
    nifti = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            up = f.upper()
            if up.endswith(".NII") or up.endswith(".NII.GZ"):
                nifti.append(os.path.join(root, f))
    return nifti


def auto_assign_sequences(nifti_list):
    """Try to auto-assign NIfTI files to FLAIR/T1/T2/MASK based on filenames.

    Returns {"FLAIR": path_or_None, "T1": ..., "T2": ..., "MASK": ...}.
    """
    results = {"FLAIR": None, "T1": None, "T2": None, "MASK": None}
    for path in nifti_list:
        up = os.path.basename(path).upper()
        if results["FLAIR"] is None and "FLAIR" in up:
            results["FLAIR"] = path
        elif results["MASK"] is None and any(k in up for k in ("MASK", "SEG", "LABEL", "WMH")):
            results["MASK"] = path
        elif results["T2"] is None and "T2" in up:
            results["T2"] = path
        elif results["T1"] is None and ("T1" in up or "3DT1" in up):
            results["T1"] = path

    # nnUNet channel convention fallback: *_0000 = FLAIR, *_0001 = T1, *_0002 = T2
    if results["FLAIR"] is None or results["T1"] is None:
        for path in nifti_list:
            base = os.path.basename(path).upper()
            if results["FLAIR"] is None and "_0000" in base:
                results["FLAIR"] = path
            elif results["T1"] is None and "_0001" in base:
                results["T1"] = path
            elif results["T2"] is None and "_0002" in base:
                results["T2"] = path

    return results


def write_nifti(arr3d, path, ref_image=None, spacing=None):
    """Write a 3D numpy array as NIfTI."""
    if _HAS_SITK:
        img = sitk.GetImageFromArray(arr3d)
        if ref_image is not None and isinstance(ref_image, sitk.Image):
            img.SetSpacing(ref_image.GetSpacing())
            img.SetOrigin(ref_image.GetOrigin())
            img.SetDirection(ref_image.GetDirection())
        elif spacing is not None:
            img.SetSpacing(spacing)
        sitk.WriteImage(img, path, useCompression=True)
    elif _HAS_NIB:
        affine = np.eye(4)
        if spacing is not None:
            affine[0, 0], affine[1, 1], affine[2, 2] = spacing
        img = nib.Nifti1Image(np.transpose(arr3d, (2, 1, 0)), affine)
        nib.save(img, path)
    else:
        raise RuntimeError("SimpleITK or nibabel required to write NIfTI files.")
