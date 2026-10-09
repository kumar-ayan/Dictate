class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(4096);
    this.offset = 0;
    this.port.onmessage = ({ data }) => {
      if (data === "flush") {
        this.sendChunk();
        this.port.postMessage({ type: "flushed" });
      }
    };
  }

  sendChunk() {
    if (!this.offset) return;
    const chunk = this.buffer.slice(0, this.offset);
    this.port.postMessage({ type: "chunk", samples: chunk.buffer }, [chunk.buffer]);
    this.buffer = new Float32Array(4096);
    this.offset = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    if (input) {
      let read = 0;
      while (read < input.length) {
        const count = Math.min(input.length - read, this.buffer.length - this.offset);
        this.buffer.set(input.subarray(read, read + count), this.offset);
        read += count;
        this.offset += count;
        if (this.offset === this.buffer.length) this.sendChunk();
      }
    }
    for (const channel of outputs[0] || []) channel.fill(0);
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
