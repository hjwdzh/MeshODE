"""ShapeNet deformation dataloader"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import trimesh
import glob
import warnings


synset_to_cat = {
    '02691156': 'airplane',
    '02933112': 'cabinet',
    '03001627': 'chair',
    '03636649': 'lamp',
    '04090263': 'rifle',
    '04379243': 'table',
    '04530566': 'watercraft',
    '02828884': 'bench',
    '02958343': 'car',
    '03211117': 'display',
    '03691459': 'speaker',
    '04256520': 'sofa',
    '04401088': 'telephone'
}

cat_to_synset = {value:key for key, value in synset_to_cat.items()}


class ShapeNetBase(Dataset):
    """Pytorch Dataset base for loading ShapeNet shape pairs.
    """
    def __init__(self, data_root, split, category="chair"):
        """
        Initialize DataSet
        Args:
          data_root: str, path to data root that contains the ShapeNet dataset.
          split: str, one of 'train'/'val'/'test'.
          catetory: str, name of the category to train on. 'all' for all 13 classes.
                    Otherwise can be a comma separated string containing multiple names.
        """
        self.data_root = data_root
        self.split = split
        splits = ['train', 'test', 'val']
        if not (split in splits):
            raise ValueError(f"{split} must be one of {splits}")
        self.categories = [c.strip() for c in category.split(',')]
        cats = list(cat_to_synset.keys())
        if 'all' in self.categories:
            self.categories = cats
        for c in self.categories:
            if not c in cats:
                raise ValueError(f"{c} is not in the list of the 13 categories: {cats}")
        self.files = self._get_filenames(self.data_root, self.split, self.categories)
        
    @staticmethod
    def _get_filenames(data_root, split, categories):
        files = []
        for c in categories:
            synset_id = cat_to_synset[c]
            cat_folder = os.path.join(data_root, split, synset_id)
            if not os.path.exists(cat_folder):
                raise RuntimeError(f"Datafolder for {synset_id} ({c}) does not exist at {cat_folder}.")
            files += sorted(glob.glob(os.path.join(cat_folder, "*/*.ply")))
        return files
        
    def __len__(self):
        nfiles = len(self.files)
        return nfiles * (nfiles - 1)
    
    @property
    def n_shapes(self):
        return len(self.files)
    
    @staticmethod
    def _idx_to_combinations(idx):
        """Translate a 1d index to a pair of indices from the combinations."""
        idx = idx + 1
        i = np.ceil((-1+np.sqrt(1+8*idx)) / 2)
        j = idx - (i * (i-1)) / 2
        return int(i)-1, int(j)-1
    

class ShapeNetVertexSampler(ShapeNetBase):
    """Pytorch Dataset for sampling vertices from meshes."""
    
    def __init__(self, data_root, split, category="chair", nsamples=5000, normals=True):
        """
        Initialize DataSet
        Args:
          data_root: str, path to data root that contains the ShapeNet dataset.
          split: str, one of 'train'/'val'/'test'.
          catetory: str, name of the category to train on. 'all' for all 13 classes.
                    Otherwise can be a comma separated string containing multiple names.
          nsamples: int, number of points to sample from each mesh.
          normals: bool, whether to add normals to the point features.
        """
        super(ShapeNetVertexSampler, self).__init__(
            data_root=data_root, split=split, category=category)
        self.nsamples = nsamples
        self.normals = normals
        
    @staticmethod
    def sample_mesh(mesh_path, nsamples, normals=True):
        """Load the mesh from mesh_path and sample nsampels points from its vertices.
        
        If nsamples < number of vertices on mesh, randomly repeat some vertices as padding.
        
        Args:
          mesh_path: str, path to load the mesh from.
          nsamples: int, number of vertices to sample.
          normals: bool, whether to add normals to the point features.
        Returns:
          v_sample: np array of shape [nsamples, 3 or 6] for sampled points.
        """
        mesh = trimesh.load(mesh_path)
        v = mesh.vertices
        nv = v.shape[0]
        seq = np.random.permutation(nv)[:nsamples]
        if len(seq) < nsamples:
            seq_repeat = np.random.choice(nv, nsamples-len(seq), replace=True)
            seq = np.concatenate([seq, seq_repeat], axis=0)
        v_sample = v[seq]
        if normals:
            n_sample = mesh.vertex_normals[seq]
            v_sample = np.concatenate([v_sample, n_sample], axis=-1)
        
        return v_sample
        
    def __getitem__(self, idx):
        """Get a random pair of shapes corresponding to idx.
        Args:
          idx: int, index of the shape pair to return. must be smaller than len(self).
        Returns:
          verts_i: [npoints, 3 or 6] float tensor for point samples from the first mesh.
          verts_j: [npoints, 3 or 6] float tensor for point samples from the second mesh.
        """
        i, j = self._idx_to_combinations(idx)
        verts_i = self.sample_mesh(self.files[i], self.nsamples, self.normals)
        verts_j = self.sample_mesh(self.files[j], self.nsamples, self.normals)
        verts_i = verts_i.astype(np.float32)
        verts_j = verts_j.astype(np.float32)
        
        return verts_i, verts_j
        
        
class ShapeNetMeshLoader(ShapeNetBase):
    """Pytorch Dataset for sampling entire meshes."""
    
    def __init__(self, data_root, split, category="chair", normals=True):
        """
        Initialize DataSet
        Args:
          data_root: str, path to data root that contains the ShapeNet dataset.
          split: str, one of 'train'/'val'/'test'.
          catetory: str, name of the category to train on. 'all' for all 13 classes.
                    Otherwise can be a comma separated string containing multiple names.
        """
        super(ShapeNetMeshLoader, self).__init__(
            data_root=data_root, split=split, category=category)
        self.normals = normals
        
    def __getitem__(self, idx):
        """Get a random pair of meshes.
        Args:
          idx: int, index of the shape pair to return. must be smaller than len(self).
        Returns:
          verts_i: [#vi, 3 or 6] float tensor for vertices from the first mesh.
          faces_i: [#fi, 3 or 6] int32 tensor for faces from the first mesh.
          verts_j: [#vj, 3 or 6] float tensor for vertices from the second mesh.
          faces_j: [#fj, 3 or 6] int32 tensor for faces from the second mesh.
        """
        i, j = self._idx_to_combinations(idx)
        mesh_i = trimesh.load(self.files[i])
        mesh_j = trimesh.load(self.files[j])
        
        verts_i = mesh_i.vertices.astype(np.float32)
        faces_i = mesh_i.faces.astype(np.int32)
        
        verts_j = mesh_j.vertices.astype(np.float32)
        faces_j = mesh_j.faces.astype(np.int32)
        
        if self.normals:
            norms_i = mesh_i.vertex_normals.astype(np.float32)
            norms_j = mesh_j.vertex_normals.astype(np.float32)
            verts_i = np.concatenate([verts_i, norms_i], axis=-1)
            verts_j = np.concatenate([verts_j, norms_j], axis=-1)
        
        verts_i = torch.from_numpy(verts_i)
        faces_i = torch.from_numpy(faces_i)
        verts_j = torch.from_numpy(verts_j)
        faces_j = torch.from_numpy(faces_j)
        
        return verts_i, faces_i, verts_j, faces_j

        
if __name__ == "__main__":
    # simple test
    dataset = ShapeNetVertexSampler(data_root="/home/maxjiang/codes/ShapeDeform/data/shapenet",
                                    split='val', nsamples=5000, normals=True, category="chair")
    print(f"Number of unique combinations of shapes: {len(dataset)}")
    print(f"Number of unique shapes: {dataset.n_shapes}")
    v0, v1 = dataset[185]
    print(v0.shape)
    print(v1.shape)
    
    meshset = ShapeNetMeshLoader(data_root="/home/maxjiang/codes/ShapeDeform/data/shapenet",
                                 split='val', normals=True, category="chair")
    v0, f0, v1, f1 = meshset[185]
    print(v0.shape)
    print(f0.shape)
    print(v1.shape)
    print(f1.shape)