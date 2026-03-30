import { Tensor } from "./types/torch"

export interface AUDIO {
    waveform: Tensor;
    sample_rate: number;
}