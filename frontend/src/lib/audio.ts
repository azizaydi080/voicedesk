/** Box-filter downsampler: float samples at the device rate -> PCM16 at 16 kHz. Keeps
 * fractional position and leftovers between calls, so chunk boundaries are seamless. */
export class Downsampler {
  private readonly ratio: number;
  private leftover = new Float32Array(0);
  private pos = 0;

  constructor(inRate: number, outRate = 16000) {
    if (inRate < outRate) throw new Error(`input rate ${inRate} below ${outRate}`);
    this.ratio = inRate / outRate;
  }

  push(input: Float32Array): Int16Array {
    const data = new Float32Array(this.leftover.length + input.length);
    data.set(this.leftover);
    data.set(input, this.leftover.length);
    const out: number[] = [];
    let pos = this.pos;
    while (pos + this.ratio <= data.length) {
      const start = Math.floor(pos);
      const end = Math.max(start + 1, Math.floor(pos + this.ratio));
      let sum = 0;
      for (let i = start; i < end; i++) sum += data[i] ?? 0;
      out.push(sum / (end - start));
      pos += this.ratio;
    }
    const consumed = Math.floor(pos);
    this.leftover = data.slice(consumed);
    this.pos = pos - consumed;
    return floatToPcm16(out);
  }
}

export function floatToPcm16(samples: ArrayLike<number>): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i] ?? 0));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/** Gapless, in-order playback of encoded chunks (MP3) with instant flush for barge-in. */
export class ChunkPlayer {
  private next = 0;
  private chain: Promise<void> = Promise.resolve();
  private generation = 0;
  private readonly live = new Set<AudioBufferSourceNode>();

  constructor(
    private readonly ctx: AudioContext,
    private readonly onIdle: () => void,
  ) {}

  /** Resolves with the performance.now() timestamp at which the chunk starts playing. */
  enqueue(encoded: ArrayBuffer): Promise<number | null> {
    const gen = this.generation;
    const result = this.chain.then(async () => {
      const buffer = await this.ctx.decodeAudioData(encoded);
      if (gen !== this.generation) return null;
      const src = this.ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(this.ctx.destination);
      const at = Math.max(this.ctx.currentTime + 0.01, this.next);
      src.start(at);
      this.next = at + buffer.duration;
      this.live.add(src);
      src.onended = () => {
        this.live.delete(src);
        if (this.live.size === 0 && gen === this.generation) this.onIdle();
      };
      return performance.now() + (at - this.ctx.currentTime) * 1000;
    });
    this.chain = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  get playing(): boolean {
    return this.live.size > 0;
  }

  stop(): void {
    this.generation++;
    for (const src of this.live) {
      try {
        src.stop();
      } catch {
        /* already stopped */
      }
    }
    this.live.clear();
    this.next = 0;
  }
}
