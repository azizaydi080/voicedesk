import { describe, expect, it } from "vitest";
import { Downsampler, floatToPcm16 } from "./audio";

describe("Downsampler", () => {
  it("produces exactly one third of the samples from 48 kHz", () => {
    const d = new Downsampler(48000);
    const out = d.push(new Float32Array(4800).fill(0.5));
    expect(out.length).toBe(1600);
    expect(out[0]).toBe(Math.trunc(0.5 * 0x7fff));
  });

  it("is seamless across odd chunk sizes and non-integer ratios", () => {
    const whole = new Downsampler(44100).push(new Float32Array(44100).fill(0.25));
    const split = new Downsampler(44100);
    let total = 0;
    for (const n of [128, 77, 1000, 3, 42892]) total += split.push(new Float32Array(n).fill(0.25)).length;
    expect(total).toBe(whole.length);
    expect(Math.abs(whole.length - 16000)).toBeLessThanOrEqual(1);
  });

  it("averages rather than drops samples", () => {
    const d = new Downsampler(48000);
    const out = d.push(Float32Array.from([1, 0, -1, 1, 1, 1]));
    expect(Array.from(out)).toEqual([0, 0x7fff]);
  });
});

describe("floatToPcm16", () => {
  it("clips and scales asymmetrically", () => {
    expect(Array.from(floatToPcm16([2, -2, 0]))).toEqual([0x7fff, -0x8000, 0]);
  });
});
