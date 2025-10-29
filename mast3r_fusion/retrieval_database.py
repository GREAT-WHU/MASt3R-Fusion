import torch
import numpy as np
from mast3r.retrieval.processor import Retriever
from mast3r.retrieval.model import how_select_local

from asmk import io_helpers


class RetrievalDatabase(Retriever):
    def __init__(self, modelname, backbone=None, device="cuda"):
        super().__init__(modelname, backbone, device)

        self.ivf_builder = self.asmk.create_ivf_builder()

        self.kf_counter = 0
        self.kf_ids = []

        self.query_dtype = torch.float32
        self.query_device = device
        self.centroids = torch.from_numpy(self.asmk.codebook.centroids).to(
            device=self.query_device, dtype=self.query_dtype
        )

    # Mirrors forward_local in extract_local_features from retrieval/model.py
    def prep_features(self, backbone_feat):
        retrieval_model = self.model

        # extract_features_and_attention without the encoding!
        backbone_feat_prewhitened = retrieval_model.prewhiten(backbone_feat)
        proj_feat = retrieval_model.projector(backbone_feat_prewhitened) + (
            0.0 if not retrieval_model.residual else backbone_feat_prewhitened
        )
        attention = retrieval_model.attention(proj_feat)
        proj_feat_whitened = retrieval_model.postwhiten(proj_feat)

        # how_select_local in
        topk_features, _, _ = how_select_local(
            proj_feat_whitened, attention, retrieval_model.nfeat
        )

        return topk_features

    def update(self, frame, add_after_query, k, min_thresh=0.0):
        feat = self.prep_features(frame.feat)
        id = self.kf_counter  # Using own counter since otherwise messes up IVF

        feat_np = feat[0].cpu().numpy()  # Assumes one frame at a time!
        id_np = id * np.ones(feat_np.shape[0], dtype=np.int64)

        database_size = self.ivf_builder.ivf.n_images
        # print("Database size: ", database_size, self.kf_counter)

        # Only query if already an image
        topk_image_inds = []
        topk_codes = None  # Change this if actualy querying
        if self.kf_counter > 0:
            ranks, ranked_scores, topk_codes = self.query(feat_np, id_np)

            scores = np.empty_like(ranked_scores)
            scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores
            scores = torch.from_numpy(scores)[0]

            topk_images = torch.topk(scores, min(k, database_size))

            valid = topk_images.values > min_thresh
            topk_image_inds = topk_images.indices[valid]
            topk_image_inds = topk_image_inds.tolist()

        if add_after_query:
            self.add_to_database(feat_np, id_np, topk_codes)

        return topk_image_inds

    # The reason we need this function is becasue kernel and inverted file not defined when manually updating ivf_builder
    def query(self, feat, id):
        step_params = self.asmk.params.get("query_ivf")

        images2, ranks, scores, topk = self.accumulate_scores(
            self.asmk.codebook,
            self.ivf_builder.kernel,
            self.ivf_builder.ivf,
            feat,
            id,
            params=step_params,
        )

        return ranks, scores, topk

    def add_to_database(self, feat_np, id_np, topk_codes):
        self.add_to_ivf_custom(feat_np, id_np, topk_codes)

        # Bookkeeping
        self.kf_ids.append(id_np[0])
        self.kf_counter += 1

    def quantize_custom(self, qvecs, params):
        # Using trick for efficient distance matrix
        l2_dists = (
            torch.sum(qvecs**2, dim=1)[:, None]
            + torch.sum(self.centroids**2, dim=1)[None, :]
            - 2 * (qvecs @ self.centroids.mT)
        )
        k = params["quantize"]["multiple_assignment"]
        topk = torch.topk(l2_dists, k, dim=1, largest=False)
        return topk.indices

    def accumulate_scores(self, cdb, kern, ivf, qvecs, qimids, params):
        """Accumulate scores for every query image (qvecs, qimids) given codebook, kernel,
        inverted_file and parameters."""
        similarity_func = lambda *x: kern.similarity(*x, **params["similarity"])

        acc = []
        slices = list(io_helpers.slice_unique(qimids))
        for imid, seq in slices:
            # Calculate qvecs to centroids distance matrix (without forming diff!)
            qvecs_torch = torch.from_numpy(qvecs[seq]).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, params)
            topk_inds = topk_inds.cpu().numpy()
            quantized = (qvecs, topk_inds)

            aggregated = kern.aggregate_image(*quantized, **params["aggregate"])
            ranks, scores = ivf.search(
                *aggregated, **params["search"], similarity_func=similarity_func
            )
            acc.append((imid, ranks, scores, topk_inds))

        imids_all, ranks_all, scores_all, topk_all = zip(*acc)

        return (
            np.array(imids_all),
            np.vstack(ranks_all),
            np.vstack(scores_all),
            np.vstack(topk_all),
        )

    def add_to_ivf_custom(self, vecs, imids, topk_codes=None):
        """Add descriptors and cooresponding image ids to the IVF

        :param np.ndarray vecs: 2D array of local descriptors
        :param np.ndarray imids: 1D array of image ids
        :param bool progress: step at which update progress printing (None to disable)
        """
        ivf_builder = self.ivf_builder

        step_params = self.asmk.params.get("build_ivf")

        if topk_codes is None:
            qvecs_torch = torch.from_numpy(vecs).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, step_params)
            topk_inds = topk_inds.cpu().numpy()
        else:
            # Reuse previously calculated! Only take top 1
            # NOTE: Assuming build params multiple assignment is less than query
            k = step_params["quantize"]["multiple_assignment"]
            topk_inds = topk_codes[:, :k]

        quantized = (vecs, topk_inds, imids)

        aggregated = ivf_builder.kernel.aggregate(
            *quantized, **ivf_builder.step_params["aggregate"]
        )
        ivf_builder.ivf.add(*aggregated)
