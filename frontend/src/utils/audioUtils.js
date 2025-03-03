export const convertToWav = async (blob) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContext.decodeAudioData(reader.result, (buffer) => {
        const offlineContext = new OfflineAudioContext(1, buffer.length, buffer.sampleRate);
        const source = offlineContext.createBufferSource();
        source.buffer = buffer;
        source.connect(offlineContext.destination);
        source.start(0);
        offlineContext.startRendering().then((renderedBuffer) => {
          const wavBlob = bufferToWave(renderedBuffer, renderedBuffer.length);
          resolve(wavBlob);
        }).catch(reject);
      }, reject);
    };
    reader.onerror = reject;
    reader.readAsArrayBuffer(blob);
  });
};

const bufferToWave = (abuffer, len) => {
  const numOfChan = abuffer.numberOfChannels;
  const length = len * numOfChan * 2 + 44;
  const buffer = new ArrayBuffer(length);
  const view = new DataView(buffer);
  const channels = [];
  let offset = 0;
  let pos = 0;

  // Write WAV header
  setUint32(0x46464952); // "RIFF"
  setUint32(length - 8); // file length - 8
  setUint32(0x45564157); // "WAVE"

  setUint32(0x20746d66); // "fmt " chunk
  setUint32(16); // length = 16
  setUint16(1); // PCM (uncompressed)
  setUint16(numOfChan);
  setUint32(abuffer.sampleRate);
  setUint32(abuffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
  setUint16(numOfChan * 2); // block-align
  setUint16(16); // 16-bit (hardcoded in this demo)

  setUint32(0x61746164); // "data" - chunk
  setUint32(length - pos - 4); // chunk length

  // Write interleaved data
  for (let i = 0; i < abuffer.numberOfChannels; i++) {
    channels.push(abuffer.getChannelData(i));
  }

  while (pos < length) {
    for (let i = 0; i < numOfChan; i++) { // interleave channels
      const sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
      view.setInt16(pos, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true); // convert to PCM
      pos += 2;
    }
    offset++; // next source sample
  }

  return new Blob([buffer], { type: "audio/wav" });

  function setUint16(data) {
    view.setUint16(pos, data, true);
    pos += 2;
  }

  function setUint32(data) {
    view.setUint32(pos, data, true);
    pos += 4;
  }
};
