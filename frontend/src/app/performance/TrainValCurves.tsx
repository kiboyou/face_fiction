
export default function TrainValCurves({ loss, val_loss, acc, val_acc }: { loss: number[]; val_loss: number[]; acc: number[]; val_acc: number[] }) {
  const width = 520, height = 280, pad = 28;
  const n = Math.max(loss.length, val_loss.length, acc.length, val_acc.length);
  const toXY = (i: number, v: number, min: number, max: number) => {
    const x = (i / (n - 1)) * (width - pad * 2);
    const y = (1 - (v - min) / (max - min)) * (height - pad * 2);
    return [x, y];
  };
  const lossMin = Math.min(...loss, ...val_loss), lossMax = Math.max(...loss, ...val_loss);
  const accMin = Math.min(...acc, ...val_acc), accMax = Math.max(...acc, ...val_acc);
  const lossPath = loss.map((v, i) => {
    const [x, y] = toXY(i, v, lossMin, lossMax);
    return `${i === 0 ? "M" : "L"}${x},${y}`;
  }).join(" ");
  const valLossPath = val_loss.map((v, i) => {
    const [x, y] = toXY(i, v, lossMin, lossMax);
    return `${i === 0 ? "M" : "L"}${x},${y}`;
  }).join(" ");
  const accPath = acc.map((v, i) => {
    const [x, y] = toXY(i, v, accMin, accMax);
    return `${i === 0 ? "M" : "L"}${x},${y}`;
  }).join(" ");
  const valAccPath = val_acc.map((v, i) => {
    const [x, y] = toXY(i, v, accMin, accMax);
    return `${i === 0 ? "M" : "L"}${x},${y}`;
  }).join(" ");
  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} className="chart">
      <g transform={`translate(${pad},${pad})`}>
        <rect x={0} y={0} width={width - pad * 2} height={height - pad * 2} rx={10} ry={10} fill="#fff" stroke="rgba(2,6,23,0.06)" />
        <path d={lossPath} fill="none" stroke="var(--color-secondary)" strokeWidth={3} />
        <path d={accPath} fill="none" stroke="var(--color-primary)" strokeWidth={3} />
        <path d={valLossPath} fill="none" stroke="var(--color-accent)" strokeWidth={2} strokeDasharray="4 2" />
        <path d={valAccPath} fill="none" stroke="var(--color-success)" strokeWidth={2} strokeDasharray="4 2" />
      </g>
    </svg>
  );
}