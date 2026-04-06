"""
Extract high-quality brain surface and lesion meshes from NIfTI data.
Uses full-resolution marching cubes + quadric decimation for gyral detail.

Usage:
    python extract_brain_mesh.py --t1 /path/to/t1.nii.gz --labels /path/to/labels.nii.gz [-o output.glb]
"""
import numpy as np
import nibabel as nib
from skimage import measure
from scipy.ndimage import gaussian_filter, binary_dilation, label as ndimage_label
import trimesh
import struct
import json
import os
import argparse

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    spacing = np.sqrt(np.sum(affine[:3,:3]**2, axis=0))
    return data, affine, spacing

def voxels_to_world(verts_ijk, affine):
    """Transform marching-cubes vertices to world coordinates via affine."""
    spacing = np.sqrt(np.sum(affine[:3,:3]**2, axis=0))
    verts_vox = verts_ijk / spacing
    ones = np.ones((len(verts_vox), 1), dtype=np.float32)
    verts_hom = np.hstack([verts_vox, ones])
    verts_world = (affine @ verts_hom.T).T[:, :3]
    return verts_world.astype(np.float32)

def world_to_threejs(verts_world):
    """NIfTI RAS -> Three.js: X=Right, Y=Up(Superior), Z=Forward(Anterior)."""
    out = np.empty_like(verts_world)
    out[:, 0] = verts_world[:, 0]
    out[:, 1] = verts_world[:, 2]
    out[:, 2] = verts_world[:, 1]
    return out

def transform_normals(normals_ijk, affine):
    """Transform normals to Three.js space."""
    rot = affine[:3, :3]
    rot_inv_t = np.linalg.inv(rot).T
    normals_world = (rot_inv_t @ normals_ijk.T).T
    norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals_world /= norms
    out = np.empty_like(normals_world)
    out[:, 0] = normals_world[:, 0]
    out[:, 1] = normals_world[:, 2]
    out[:, 2] = normals_world[:, 1]
    return out.astype(np.float32)

def decimate_mesh(verts, faces, target_faces):
    """Quadric decimation using trimesh -- preserves sharp features (sulci)."""
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if len(faces) <= target_faces:
        return verts, faces
    print(f"  Decimating {len(faces)} -> {target_faces} faces...")
    # trimesh uses simplify_quadric_decimation with a face count target
    # but the underlying fast_simplification uses a 0-1 ratio
    try:
        ratio = 1.0 - (target_faces / len(faces))  # reduction ratio
        import fast_simplification
        verts_out, faces_out = fast_simplification.simplify(
            verts, faces, target_reduction=ratio, agg=7)
        print(f"  Result: {len(verts_out)} verts, {len(faces_out)} faces")
        return np.array(verts_out, dtype=np.float32), np.array(faces_out)
    except Exception as e:
        print(f"  fast_simplification failed ({e}), trying trimesh...")
        # Fallback: use a voxel-based approach
        ratio = target_faces / len(faces)
        # Subsample by taking every Nth face
        step = max(1, int(1 / ratio))
        subset_faces = faces[::step]
        used_verts = np.unique(subset_faces)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used_verts] = np.arange(len(used_verts))
        new_faces = remap[subset_faces]
        valid = np.all(new_faces >= 0, axis=1)
        return verts[used_verts].astype(np.float32), new_faces[valid]

def extract_brain_surface(data, affine, spacing, threshold_pct=30, smooth_sigma=0.5, target_faces=100000):
    """Extract brain surface with gyral detail preserved."""
    # Minimal smoothing to keep sulci visible
    smoothed = gaussian_filter(data, sigma=smooth_sigma)
    nonzero = smoothed[smoothed > 0]
    threshold = np.percentile(nonzero, threshold_pct)
    brain_mask = smoothed > threshold

    # Keep only the largest connected component
    labeled, num_features = ndimage_label(brain_mask)
    if num_features > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # ignore background
        largest = sizes.argmax()
        brain_mask = labeled == largest
        print(f"  Kept largest island ({num_features} components -> 1)")

    print(f"  Threshold: P{threshold_pct} = {threshold:.1f}")

    # Full-resolution marching cubes
    verts, faces, normals, _ = measure.marching_cubes(
        brain_mask.astype(np.float32), level=0.5, spacing=tuple(spacing))
    print(f"  Raw mesh: {len(verts)} verts, {len(faces)} faces")

    # Decimate while preserving features
    verts, faces = decimate_mesh(verts, faces, target_faces)

    # Recompute normals after decimation
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    normals = mesh.vertex_normals.astype(np.float32)

    # Transform to Three.js coordinates
    verts_world = voxels_to_world(verts, affine)
    verts_3js = world_to_threejs(verts_world)
    normals_3js = transform_normals(normals, affine)

    # Center and normalize
    center = verts_3js.mean(axis=0)
    verts_3js -= center
    max_ext = np.abs(verts_3js).max()
    verts_3js /= max_ext

    return verts_3js, faces, normals_3js, center, max_ext, affine

def extract_lesion_meshes(label_data, affine, spacing, center_3js, max_ext, dilate=1):
    """Extract lesion mesh."""
    if label_data.sum() == 0:
        return None, None, None
    mask = label_data > 0
    if dilate > 0:
        mask = binary_dilation(mask, iterations=dilate)
    smoothed = gaussian_filter(mask.astype(np.float32), sigma=0.6)
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            smoothed, level=0.3, spacing=tuple(spacing))
    except ValueError:
        return None, None, None

    # Recompute normals via trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    normals = mesh.vertex_normals.astype(np.float32)

    verts_world = voxels_to_world(verts, affine)
    verts_3js = world_to_threejs(verts_world)
    normals_3js = transform_normals(normals, affine)

    verts_3js -= center_3js
    verts_3js /= max_ext

    print(f"  Lesion mesh: {len(verts)} verts, {len(faces)} faces")
    return verts_3js, faces, normals_3js


def build_glb(meshes):
    """Build binary GLB (glTF 2.0)."""
    accessors, buffer_views, mesh_primitives, nodes = [], [], [], []
    all_bin = bytearray()

    for i, m in enumerate(meshes):
        verts = m['vertices'].astype(np.float32)
        norms = m['normals'].astype(np.float32)
        faces = m['faces'].astype(np.uint32)

        for data_bytes, target in [(verts.tobytes(), 34962), (norms.tobytes(), 34962), (faces.tobytes(), 34963)]:
            offset = len(all_bin)
            all_bin.extend(data_bytes)
            while len(all_bin) % 4 != 0: all_bin.append(0)
            buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data_bytes), "target": target})

        bv = len(buffer_views) - 3
        acc = len(accessors)
        accessors.append({"bufferView": bv, "componentType": 5126, "count": len(verts), "type": "VEC3",
                          "min": verts.min(axis=0).tolist(), "max": verts.max(axis=0).tolist()})
        accessors.append({"bufferView": bv+1, "componentType": 5126, "count": len(norms), "type": "VEC3"})
        accessors.append({"bufferView": bv+2, "componentType": 5125, "count": faces.size, "type": "SCALAR"})
        mesh_primitives.append({"primitives": [{"attributes": {"POSITION": acc, "NORMAL": acc+1}, "indices": acc+2, "material": i}], "name": m['name']})
        nodes.append({"mesh": i, "name": m['name']})

    materials = []
    for m in meshes:
        c = m['color']
        mat = {"name": m['name'], "pbrMetallicRoughness": {"baseColorFactor": c[:4] if len(c)>=4 else c+[1.0], "metallicFactor": 0.0, "roughnessFactor": 0.6}}
        if len(c)>3 and c[3]<1.0: mat["alphaMode"] = "BLEND"
        if m.get('emissive'): mat["emissiveFactor"] = m['emissive']
        materials.append(mat)

    gltf = {"asset": {"version": "2.0", "generator": "MSLesionTool"}, "scene": 0,
            "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
            "meshes": mesh_primitives, "materials": materials,
            "accessors": accessors, "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(all_bin)}]}

    gltf_json = json.dumps(gltf, separators=(',',':')).encode('utf-8')
    while len(gltf_json) % 4 != 0: gltf_json += b' '

    total = 12 + 8 + len(gltf_json) + 8 + len(all_bin)
    glb = bytearray(b'glTF')
    glb.extend(struct.pack('<III', 2, total, len(gltf_json)))
    glb.extend(struct.pack('<I', 0x4E4F534A))
    glb.extend(gltf_json)
    glb.extend(struct.pack('<I', len(all_bin)))
    glb.extend(struct.pack('<I', 0x004E4942))
    glb.extend(all_bin)
    return bytes(glb)


def main():
    parser = argparse.ArgumentParser(description="Extract brain surface + lesion meshes to GLB")
    parser.add_argument("--t1", required=True, help="Path to T1 NIfTI (e.g. sub_0001.nii.gz)")
    parser.add_argument("--labels", required=True, help="Path to lesion label NIfTI")
    parser.add_argument("-o", "--output", default=None, help="Output GLB path (default: brain-splash-preview-brain.glb in script dir)")
    args = parser.parse_args()

    t1_path = args.t1
    label_path = args.labels

    print(f"Loading T1: {t1_path}")
    t1, affine, spacing = load_nifti(t1_path)
    print(f"  Shape: {t1.shape}, Spacing: {spacing}, Orient: {nib.aff2axcodes(affine)}")

    print(f"Loading labels: {label_path}")
    labels, _, _ = load_nifti(label_path)
    print(f"  Lesion voxels: {int((labels > 0).sum())}")

    # Extract brain surface: low smoothing for sulci detail, 100K face target
    print("\nExtracting brain surface (high-detail)...")
    brain_v, brain_f, brain_n, center, scale, used_aff = extract_brain_surface(
        t1, affine, spacing, threshold_pct=28, smooth_sigma=0.5, target_faces=100000)

    print("\nExtracting lesion meshes...")
    les_v, les_f, les_n = extract_lesion_meshes(
        labels, used_aff, spacing, center, scale, dilate=1)

    meshes = [{'name': 'brain', 'vertices': brain_v, 'normals': brain_n, 'faces': brain_f,
               'color': [0.808, 0.753, 0.659, 1.0]}]
    if les_v is not None:
        meshes.append({'name': 'lesions', 'vertices': les_v, 'normals': les_n, 'faces': les_f,
                       'color': [1.0, 0.2, 0.35, 0.9], 'emissive': [0.8, 0.1, 0.2]})

    print("\nBuilding GLB...")
    glb = build_glb(meshes)
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain-splash-preview-brain.glb")
    with open(out, 'wb') as f:
        f.write(glb)
    print(f"Saved: {out} ({len(glb)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
