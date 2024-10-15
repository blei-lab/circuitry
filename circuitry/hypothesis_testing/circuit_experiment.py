from circuitry.circuit import Circuit, EdgeLevelMaskedTransformer


class CircuitExperiment:
    def __init__(
        self,
        ref_ds,
        tl_model,
        zero_ablation: bool,
        use_pos_embed,
        verbose,
    ):
        self.ref_ds = ref_ds
        self.tl_model = tl_model
        self.zero_ablation = zero_ablation
        self.use_pos_embed = use_pos_embed
        self.verbose = verbose

        # Note: masked_model is initialized to a random-ish circuit rather than
        # the complete circuit, so must call `set_circuit` before `forward`.
        self.masked_model = EdgeLevelMaskedTransformer(
            tl_model,
            mask_init_p=1.0,
            starting_point_type="pos_embed" if use_pos_embed else "resid_pre",
            verbose=verbose,
        )

    def set_circuit(self, circuit: Circuit, invert: bool = False):
        self.masked_model.set_binary_mask(circuit, invert=invert)

    def get_circuit(self):
        return self.masked_model.sample_circuit_from_mask()

    def forward(self, ds, return_type="logits", loss_per_token=False):
        patch_ds = None if self.zero_ablation else self.ref_ds
        self.masked_model.calculate_and_store_ablation_cache(patch_ds)

        with self.masked_model.with_fwd_hooks_and_new_ablation_cache(patch_ds) as hooked_model:
            result = hooked_model(ds, return_type=return_type, loss_per_token=loss_per_token)

        return result
