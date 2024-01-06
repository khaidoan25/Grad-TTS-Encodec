import math
import typing as tp

from academicodec.models.encodec.net3 import SoundStream

from encodec.model import EncodecModel, EncodedFrame
from encodec.modules.seanet import SEANetDecoder, SEANetEncoder
from encodec.quantization.vq import ResidualVectorQuantizer
import encodec.modules as m
import encodec.quantization as qt

import torch
from pathlib import Path

class EncodecModelExtend(EncodecModel):
    def __init__(self, encoder: SEANetEncoder, decoder: SEANetDecoder, quantizer: ResidualVectorQuantizer, target_bandwidths: tp.List[float], sample_rate: int, channels: int, normalize: bool = False, segment: float | None = None, overlap: float = 0.01, name: str = 'unset'):
        super().__init__(encoder, decoder, quantizer, target_bandwidths, sample_rate, channels, normalize, segment, overlap, name)
        
    @staticmethod
    def _get_model(target_bandwidths: tp.List[float],
                   sample_rate: int = 24_000,
                   channels: int = 1,
                   causal: bool = True,
                   model_norm: str = 'weight_norm',
                   audio_normalize: bool = False,
                   segment: tp.Optional[float] = None,
                   name: str = 'unset'):
        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal)
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )
        model = EncodecModelExtend(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return model
        
    @staticmethod
    def encodec_model_24khz(pretrained: bool = True, repository: tp.Optional[Path] = None):
        """Return the pretrained causal 24khz model.
        """
        if repository:
            assert pretrained
        target_bandwidths = [1.5, 3., 6, 12., 24.]
        checkpoint_name = 'encodec_24khz-d7cc33bc.th'
        sample_rate = 24_000
        channels = 1
        model = EncodecModelExtend._get_model(
            target_bandwidths, sample_rate, channels,
            causal=True, model_norm='weight_norm', audio_normalize=False,
            name='encodec_24khz' if pretrained else 'unset')
        if pretrained:
            state_dict = EncodecModelExtend._get_pretrained(checkpoint_name, repository)
            model.load_state_dict(state_dict)
        model.eval()
        return model
        
    def get_emb(self, x: torch.Tensor) -> tp.List[torch.Tensor]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each embeding is has the shape `[B, K, T]`, with `K` the embedding dimension.
        """
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None
            
        emb_frames: tp.List[torch.Tensor] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset: offset + segment_length]
            length = frame.shape[-1]
            duration = length / self.sample_rate
            assert self.segment is None or duration <= 1e-5 + self.segment

            emb = self.encoder(frame)
            emb_frames.append(emb)
            
        return emb_frames[0]
    
    def encode_emb(self, x: torch.Tensor) -> torch.Tensor:
        emb = x
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes
    
class Encodec16KHz(SoundStream):
    def __init__(self, n_filters=32, D=512, target_bandwidths=[1, 1.5, 2, 4, 6, 12], ratios=[8, 5, 4, 2], sample_rate=16_000, bins=1024, normalize=False):
        super().__init__(n_filters, D, target_bandwidths, ratios, sample_rate, bins, normalize)
        self.sample_rate = sample_rate
        self.channels = 1
        
    def get_emb(self, x):
        e = self.encoder(x)
        
        return e
    
    def encode_emb(self, e, target_bw=6, st=None):
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        if st is None:
            st = 0
        codes = self.quantizer.encode(e, self.frame_rate, bw, st)
        return codes