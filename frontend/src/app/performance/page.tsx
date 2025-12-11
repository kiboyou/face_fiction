"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import { fetchMetrics } from "./api";


function SingleCurve({ data, color, label }: { data: number[]; color: string; label: string }) {
	const width = 520, height = 280, pad = 44;
	const n = data.length;
	if (!n) return null;
	const min = Math.min(...data), max = Math.max(...data);
	const path = data.map((v, i) => {
		const x = (i / (n - 1)) * (width - pad * 2);
		const y = (1 - (v - min) / (max - min)) * (height - pad * 2);
		return `${i === 0 ? "M" : "L"}${x},${y}`;
	}).join(" ");
	// Add grid lines (règles)
	const gridSteps = [0, 0.25, 0.5, 0.75, 1];
	const gridLines = gridSteps.map((t) => (
		<g key={"y-"+t}>
			<line
				x1={0}
				y1={(1 - t) * (height - pad * 2)}
				x2={width - pad * 2}
				y2={(1 - t) * (height - pad * 2)}
				stroke="#e5e7eb"
				strokeDasharray="2 2"
				strokeWidth={1}
			/>
			<text
				x={-20}
				y={(1 - t) * (height - pad * 2) + 4}
				fontSize={12}
				fill="#64748b"
				textAnchor="end"
			>
				{(min + (max - min) * t).toFixed(2)}
			</text>
		</g>
	));
	const xSteps = [0, 0.25, 0.5, 0.75, 1];
	const xLabels = xSteps.map((t) => (
		<text
			key={"x-"+t}
			x={t * (width - pad * 2)}
			y={(height - pad * 2) + 20}
			fontSize={12}
			fill="#64748b"
			textAnchor="middle"
		>
			{Math.round(t * (n - 1))}
		</text>
	));
	return (
		<svg width="100%" viewBox={`0 0 ${width} ${height + 24}`} className="chart">
			<g transform={`translate(${pad},${pad})`}>
				<rect x={0} y={0} width={width - pad * 2} height={height - pad * 2} rx={10} ry={10} fill="#fff" stroke="rgba(2,6,23,0.06)" />
				{gridLines}
				{xLabels}
				<path d={path} fill="none" stroke={color} strokeWidth={3} />
			</g>
		</svg>
	);
}

export default function PerformancePage() {
	
	const [metrics, setMetrics] = useState<any>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const [models, setModels] = useState<Array<{ name: string; type: string }>>([]);
	const [selectedModel, setSelectedModel] = useState<string>("default");

	useEffect(() => {
		let mounted = true;
		async function loadModels() {
			try {
				const data = await import("./api_models").then(mod => mod.fetchModels());
				if (mounted && Array.isArray(data.models)) {
					setModels(data.models);
				}
			} catch {
				setModels([]);
			}
		}
		loadModels();
		return () => { mounted = false; };
	}, []);

	useEffect(() => {
		let mounted = true;
		async function loadMetrics() {
			setLoading(true);
			setError(null);
			try {
				const data = await fetchMetrics(selectedModel);
				if (mounted) setMetrics(data);
			} catch (e: any) {
				if (mounted) setError("Impossible de charger les métriques du modèle.");
			} finally {
				if (mounted) setLoading(false);
			}
		}
		loadMetrics();
		return () => { mounted = false; };
	}, [selectedModel]);

	// Données dynamiques : tout vient du backend si possible (adapté au nouveau format JSON)
	const perf = useMemo(() => {
		const eval_ = metrics?.evaluation ?? {};
		const curves = eval_.curves ?? {};
		const roc = eval_.roc_curve ?? {};
		const pr = eval_.pr_curve ?? {};
		const cm = eval_.confusion_matrix ?? null;
		const dataset = eval_.dataset_info ?? {};
		
		return {
			train: {
				accuracy: eval_.train?.accuracy ?? metrics?.best_train_accuracy ?? null,
				precision: eval_.train?.precision ?? null,
				recall: eval_.train?.recall ?? null,
				f1: eval_.train?.f1 ?? null,
				auc: eval_.train?.auc ?? metrics?.best_train_auc ?? null,
				auc_pr: eval_.train?.auc_pr ?? null,
				loss: metrics?.final_train_loss ?? (curves.loss ? curves.loss[curves.loss.length-1] : null),
			},
			val: {
				accuracy: eval_.validation?.accuracy ?? metrics?.best_val_accuracy ?? null,
				precision: eval_.validation?.precision ?? null,
				recall: eval_.validation?.recall ?? null,
				f1: eval_.validation?.f1 ?? null,
				auc: eval_.validation?.auc ?? metrics?.best_val_auc ?? null,
				auc_pr: eval_.validation?.auc_pr ?? null,
				loss: metrics?.final_val_loss ?? (curves.val_loss ? curves.val_loss[curves.val_loss.length-1] : null),
			},
			latency_ms: eval_.latency_ms ?? null,
			classes: dataset.classes ?? ["Real", "Fake"],
			cm: cm,
			roc: (roc.fpr && roc.tpr) ? roc.fpr.map((fpr: number, i: number) => [fpr, roc.tpr[i]]) : null,
			pr: (pr.precision && pr.recall) ? pr.precision.map((p: number, i: number) => [pr.recall[i], p]) : null,
			loss: curves.loss ?? null,
			val_loss: curves.val_loss ?? null,
			acc: curves.accuracy ?? null,
			val_acc: Array.isArray(curves.val_accuracy) && curves.val_accuracy.length > 0 ? curves.val_accuracy : null,
			model: metrics?.model ?? "LSTM Multimodal"
		};
	}, [metrics]);

	return (
		<section className="section section-soft-bg">
			<div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
				<div className="section-title">
					<span className="tag-accent">Performance</span>
					<h1 className="mt-3 text-3xl md:text-4xl lg:text-5xl font-extrabold tracking-tight">Évaluation du modèle</h1>
					<div className="title-underline" />
						<div className="mt-4">
							<span className="block text-sm font-medium mb-1">Choisir le modèle :</span>
							<select
								className="input input-bordered w-full max-w-xs"
								value={selectedModel}
								onChange={e => setSelectedModel(e.target.value)}
								disabled={loading}
							>
								<option value="default">Auto (par défaut)</option>
								{models.map((m) => (
									<option key={m.name} value={m.name}>{m.name} ({m.type})</option>
								))}
							</select>
						</div>
					{loading && <p className="mt-3 text-slate-600 max-w-2xl mx-auto text-center">Chargement des métriques…</p>}
					{error && <p className="mt-3 text-red-600 max-w-2xl mx-auto text-center">{error}</p>}
				</div>

				{/* KPI cards - train/val */}
				<div className="mt-8 kpi-grid">
					<div className="kpi-card">
						<div className="kpi-label">ROC‑AUC (Train)</div>
						<div className="kpi-value">{perf.train.auc !== null ? (perf.train.auc * 100).toFixed(1) + "%" : "-"}</div>
						<div className="kpi-sub">Surface sous la courbe</div>
					</div>
					<div className="kpi-card">
						<div className="kpi-label">ROC‑AUC (Val)</div>
						<div className="kpi-value">{perf.val.auc !== null ? (perf.val.auc * 100).toFixed(1) + "%" : "-"}</div>
						<div className="kpi-sub">Surface sous la courbe</div>
					</div>
					<div className="kpi-card">
						<div className="kpi-label">Accuracy (Train)</div>
						<div className="kpi-value">{perf.train.accuracy !== null ? (perf.train.accuracy * 100).toFixed(1) + "%" : "-"}</div>
						<div className="kpi-sub">Exactitude globale</div>
					</div>
					<div className="kpi-card">
						<div className="kpi-label">Accuracy (Val)</div>
						<div className="kpi-value">{perf.val.accuracy !== null ? (perf.val.accuracy * 100).toFixed(1) + "%" : "-"}</div>
						<div className="kpi-sub">Exactitude globale</div>
					</div>
					<div className="kpi-card">
						<div className="kpi-label">F1‑Score (Train)</div>
						<div className="kpi-value">{perf.train.f1 !== null ? (perf.train.f1 * 100).toFixed(1) + "%" : "-"}</div>
						<div className="kpi-sub">Harmonique Précision/Rappel</div>
					</div>
					<div className="kpi-card">
						<div className="kpi-label">F1‑Score (Val)</div>
						<div className="kpi-value">{perf.val.f1 !== null ? (perf.val.f1 * 100).toFixed(1) + "%" : "-"}</div>
						<div className="kpi-sub">Harmonique Précision/Rappel</div>
					</div>
					<div className="kpi-card">
						<div className="kpi-label">Latence</div>
						<div className="kpi-value">{perf.latency_ms !== null ? perf.latency_ms.toFixed(1).replace(/\B(?=(\d{3})+(?!\d))/g, "\u202F") + " ms" : "-"}</div>
						<div className="kpi-sub">Temps d'inférence</div>
					</div>
				</div>

				{/* Charts row 1: ROC + Confusion Matrix */}
				<div className="mt-8 grid md:grid-cols-2 gap-6">
					<div className="chart-card">
						<div className="chart-title">Courbe ROC</div>
						{perf.roc ? (
							<RocChart points={perf.roc as [number, number][]} />
						) : (
							<div className="text-slate-400 text-center py-12">Aucune donnée ROC</div>
						)}
						<div className="legend">
							<span className="dot dot-primary" /> Modèle
							<span className="sep" /> Diagonale aléatoire
						</div>
					</div>
					<div className="chart-card">
						<div className="chart-title">Matrice de confusion</div>
						{perf.cm ? (
							<ConfusionMatrix cm={perf.cm} labels={perf.classes} />
						) : (
							<div className="text-slate-400 text-center py-12">Aucune donnée</div>
						)}
					</div>
				</div>

				{/* Charts row 2: Loss/Accuracy (Train/Val) separated + PR */}
				<div className="mt-6 grid md:grid-cols-2 gap-6">
					<div className="chart-card">
						<div className="chart-title">Loss (Train)</div>
						{perf.loss ? <SingleCurve data={perf.loss} color="var(--color-secondary)" label="Loss (Train)" /> : <div className="text-slate-400 text-center py-12">Aucune courbe</div>}	
					</div>
					<div className="chart-card">
						<div className="chart-title">Accuracy (Train)</div>
						{perf.acc ? <SingleCurve data={perf.acc} color="var(--color-primary)" label="Accuracy (Train)" /> : <div className="text-slate-400 text-center py-12">Aucune courbe</div>}
					</div>
					<div className="chart-card">
						<div className="chart-title">Loss (Val)</div>
						{perf.val_loss ? <SingleCurve data={perf.val_loss} color="var(--color-accent)" label="Loss (Val)" /> : <div className="text-slate-400 text-center py-12">Aucune courbe</div>}
					</div>
					<div className="chart-card">
						<div className="chart-title">Accuracy (Val)</div>
						{perf.val_acc ? (
							<SingleCurve data={perf.val_acc} color="#f97316" label="Accuracy (Val)" />
						) : (
							<div className="text-slate-400 text-center py-12">
								Aucune courbe<br />
								<span className="text-xs">(val_accuracy manquant ou vide)</span>
							</div>
						)}
					</div>
					<div className="chart-card">
						<div className="chart-title">Courbe Précision‑Rappel</div>
						{perf.pr ? (
							<PrChart points={perf.pr as [number, number][]} />
						) : (
							<div className="text-slate-400 text-center py-12">Aucune donnée PR</div>
						)}
						<div className="legend">
							<span className="dot dot-primary" /> Modèle
						</div>
					</div>
				</div>

				{/* Architecture & détails - design amélioré */}
				<div className="mt-10 card p-6 bg-linear-to-br from-white to-slate-50 border border-violet-100 shadow-lg">
					<h2 className="font-bold text-2xl mb-6 flex items-center gap-3">
						Architecture & Détails
					</h2>
					<div className="grid md:grid-cols-2 gap-6">
						<div className="space-y-4">
							<div className="flex items-center gap-3">
								<span className="inline-block bg-violet-100 text-violet-700 px-2 py-1 rounded font-mono text-xs">Modèle</span>
								<span className="font-semibold text-lg">LSTM Multimodal</span>
							</div>
							<div className="flex items-center gap-3">
								<span className="inline-block bg-blue-100 text-blue-700 px-2 py-1 rounded font-mono text-xs">Entrées vidéo</span>
								<span className="font-mono text-base">{metrics?.input_shapes?.video_input ?? "-"}</span>
							</div>
							<div className="flex items-center gap-3">
								<span className="inline-block bg-cyan-100 text-cyan-700 px-2 py-1 rounded font-mono text-xs">Entrées audio</span>
								<span className="font-mono text-base">{metrics?.input_shapes?.audio_input ?? "-"}</span>
							</div>
							<div className="flex items-center gap-3">
								<span className="inline-block bg-slate-200 text-slate-700 px-2 py-1 rounded font-mono text-xs">Horodatage</span>
								<span className="font-mono text-base">{metrics?.timestamp ?? "-"}</span>
							</div>
						</div>
						<div className="space-y-4">
							<div className="flex items-center gap-3">
								<span className="inline-block bg-emerald-100 text-emerald-700 px-2 py-1 rounded font-mono text-xs">Perte finale</span>
								<span className="font-mono text-base">{metrics?.final_val_loss ? metrics.final_val_loss.toFixed(3) : "-"}</span>
							</div>
						<div className="flex items-center gap-3">
							<span className="inline-block bg-orange-100 text-orange-700 px-2 py-1 rounded font-mono text-xs">Train</span>
							<span className="font-mono text-base">{metrics?.evaluation?.dataset_info?.train_samples ?? "-"}</span>
						</div>
						<div className="flex items-center gap-3">
							<span className="inline-block bg-pink-100 text-pink-700 px-2 py-1 rounded font-mono text-xs">Validation</span>
							<span className="font-mono text-base">{metrics?.evaluation?.dataset_info?.val_samples ?? "-"}</span>
						</div>
						</div>
					</div>
				</div>
			</div>
		</section>
	);
}

function svgPath(points: [number, number][], width: number, height: number) {
	const toXY = (p: [number, number]) => {
		const [x, y] = p;
		return [x * width, (1 - y) * height];
	};
	return points
		.map((p, i) => {
			const [x, y] = toXY(p);
			return `${i === 0 ? "M" : "L"}${x},${y}`;
		})
		.join(" ");
}

function RocChart({ points }: { points: [number, number][] }) {
	const width = 520, height = 280, pad = 28;
	const path = svgPath(points, width - pad * 2, height - pad * 2);
	return (
		<svg width="100%" viewBox={`0 0 ${width} ${height}`} className="chart">
			<g transform={`translate(${pad},${pad})`}>
				<rect x={0} y={0} width={width - pad * 2} height={height - pad * 2} rx={10} ry={10} fill="#fff" stroke="rgba(2,6,23,0.06)" />
				<line x1={0} y1={height - pad * 2} x2={width - pad * 2} y2={0} stroke="#e5e7eb" strokeDasharray="4 4" />
				<path d={path} fill="none" stroke="var(--color-primary)" strokeWidth={3} />
				<Axis width={width - pad * 2} height={height - pad * 2} />
			</g>
		</svg>
	);
}

function PrChart({ points }: { points: [number, number][] }) {
	const width = 520, height = 280, pad = 28;
	const path = svgPath(points, width - pad * 2, height - pad * 2);
	return (
		<svg width="100%" viewBox={`0 0 ${width} ${height}`} className="chart">
			<g transform={`translate(${pad},${pad})`}>
				<rect x={0} y={0} width={width - pad * 2} height={height - pad * 2} rx={10} ry={10} fill="#fff" stroke="rgba(2,6,23,0.06)" />
				<path d={path} fill="none" stroke="var(--color-primary)" strokeWidth={3} />
				<Axis width={width - pad * 2} height={height - pad * 2} />
			</g>
		</svg>
	);
}

function TrainCurves({ loss, acc }: { loss: number[]; acc: number[] }) {
	const width = 520, height = 280, pad = 28;
	const n = Math.max(loss.length, acc.length);
	const toXY = (i: number, v: number, min: number, max: number) => {
		const x = (i / (n - 1)) * (width - pad * 2);
		const y = (1 - (v - min) / (max - min)) * (height - pad * 2);
		return [x, y];
	};
	const lossMin = Math.min(...loss), lossMax = Math.max(...loss);
	const accMin = Math.min(...acc), accMax = Math.max(...acc);
	const lossPath = loss
		.map((v, i) => {
			const [x, y] = toXY(i, v, lossMin, lossMax);
			return `${i === 0 ? "M" : "L"}${x},${y}`;
		})
		.join(" ");
	const accPath = acc
		.map((v, i) => {
			const [x, y] = toXY(i, v, accMin, accMax);
			return `${i === 0 ? "M" : "L"}${x},${y}`;
		})
		.join(" ");
	return (
		<svg width="100%" viewBox={`0 0 ${width} ${height}`} className="chart">
			<g transform={`translate(${pad},${pad})`}>
				<rect x={0} y={0} width={width - pad * 2} height={height - pad * 2} rx={10} ry={10} fill="#fff" stroke="rgba(2,6,23,0.06)" />
				<path d={lossPath} fill="none" stroke="var(--color-secondary)" strokeWidth={3} />
				<path d={accPath} fill="none" stroke="var(--color-primary)" strokeWidth={3} />
				<Axis width={width - pad * 2} height={height - pad * 2} />
			</g>
		</svg>
	);
}

function Axis({ width, height }: { width: number; height: number }) {
	return (
		<g>
			<line x1={0} y1={height} x2={width} y2={height} stroke="#e5e7eb" />
			<line x1={0} y1={0} x2={0} y2={height} stroke="#e5e7eb" />
			{[0, 0.5, 1].map((t) => (
				<g key={t}>
					<line x1={t * width} y1={height} x2={t * width} y2={height - 6} stroke="#94a3b8" />
					<text x={t * width} y={height + 16} fontSize={10} fill="#64748b" textAnchor="middle">{t}</text>
					<line x1={0} y1={(1 - t) * height} x2={6} y2={(1 - t) * height} stroke="#94a3b8" />
					<text x={-10} y={(1 - t) * height + 3} fontSize={10} fill="#64748b" textAnchor="end">{t}</text>
				</g>
			))}
		</g>
	);
}

function ConfusionMatrix({ cm, labels }: { cm: number[][]; labels: string[] }) {
	const n = cm.length;
	const total = cm.flat().reduce((a, b) => a + b, 0);
	const max = Math.max(...cm.flat());
	return (
		<div>
			<div className="cm-grid" style={{ gridTemplateColumns: `repeat(${n + 1}, minmax(0, 1fr))` }}>
				<div />
				{labels.map((l) => (
					<div key={`col-${l}`} className="cm-h">Préd: {l}</div>
				))}
				{cm.map((row, i) => (
					<Fragment key={`row-${i}`}>
						<div key={`rowh-${i}`} className="cm-h">Vrai: {labels[i]}</div>
						{row.map((v, j) => {
							const intensity = v / max;
							return (
								<div key={`cell-${i}-${j}`} className="cm-cell" style={{ background: `rgba(0,123,255,${0.08 + intensity * 0.28})` }}>
									<div className="cm-v">{v}</div>
									<div className="cm-p">{((v / total) * 100).toFixed(1)}%</div>
								</div>
							);
						})}
					</Fragment>
				))}
			</div>
			<div className="legend mt-2"><span className="dot dot-primary" /> Intensité = fréquence</div>
		</div>
	);
}
