from models.bat import BAT
from models.head.xcorr import BoxAwareXCorr


class CMT(BAT):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        self.xcorr = BoxAwareXCorr(
            feature_channel=self.config.feature_channel,
            hidden_channel=self.config.hidden_channel,
            out_channel=self.config.out_channel,
            k=self.config.k,
            use_search_bc=self.config.use_search_bc,
            use_search_feature=self.config.use_search_feature,
            bc_channel=self.config.bc_channel,
            scf_oe_use_xyz=self.config.scf_oe_use_xyz,
            neighbor_method=self.config.neighbor_method,
            kc=self.config.kc,
            scf_match_method=self.config.scf_match_method,
            scf_swin_window_sizes=self.config.swin_window_sizes,
            k2=self.config.k2,
            attn_pe=self.config.attn_pe,
            oe_radius=self.config.oe_radius,
        )
