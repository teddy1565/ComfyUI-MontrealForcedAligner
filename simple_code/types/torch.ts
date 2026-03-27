export interface Size {

}

/**
 * Batch
 * 
 * Each Chunk how many batch
 */
export type Batch = number;

/**
 * Audio Channels
 * 
 * 1 = single
 * 2 = stereo
 */
export type Channels = number;

/**
 * Time/Sample
 * 
 * How many Samples
 */
export type Time = number;

export interface Tensor {
    squeeze(): Tensor;
    shape: [Batch, Channels, Time];
}