import { useState } from 'react';
import { useApi, useJobPoller } from './hooks';
import { Sidebar } from './Sidebar';
import { JobMonitor } from './JobMonitor';

type SourceType = 'path' | 'hub';

function App() {
  const { apiReady, health, quantize, job } = useApi();

  const [sourceType, setSourceType] = useState<SourceType>('path');
  const [modelPath, setModelPath] = useState('');
  const [repoId, setRepoId] = useState('');
  const [revision, setRevision] = useState('main');
  const [algorithm, setAlgorithm] = useState('adaptive');
  const [granularity, setGranularity] = useState('per_channel');
  const [groupSize, setGroupSize] = useState(32);
  const [calibrate, setCalibrate] = useState(false);
  const [quantizeEmbeddings, setQuantizeEmbeddings] = useState(false);
  const [quantizeFinalNorm, setQuantizeFinalNorm] = useState(false);
  const [quantizeBias, setQuantizeBias] = useState(false);
  const [format, setFormat] = useState('gguf');
  const [target, setTarget] = useState('mobile');
  const [device, setDevice] = useState('auto');
  const [layerInclude, setLayerInclude] = useState('');
  const [layerExclude, setLayerExclude] = useState('');
  const [optimizeInference, setOptimizeInference] = useState(false);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ percent: 0, text: '', logs: [] as Array<{ type: string; msg: string; time: string }> });
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  useJobPoller(jobId, (job: any) => {
    if (job.status === 'completed') {
      setResult(job.result);
      setProgress(p => ({ ...p, percent: 100, text: 'Completed' }));
      addLog('success', `Done! Compression: ${(job.result?.compression_ratio ?? 0).toFixed(1)}x`);
      setRunning(false);
    } else if (job.status === 'failed') {
      setError(job.error || 'Quantization failed');
      addLog('error', job.error || 'Quantization failed');
      setRunning(false);
    } else {
      const percent = job.status === 'running' ? 65 : job.status === 'queued' ? 20 : 50;
      setProgress(p => ({ ...p, percent, text: job.status === 'queued' ? 'Queued...' : 'Running...' }));
    }
  });

  const cancelJob = async () => {
    if (!jobId) return;
    try {
      await fetch(`${API_BASE}/api/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
    } catch {}
    setRunning(false);
    addLog('warn', 'Cancel requested.');
  };

  const addLog = (type: string, msg: string) => {
    setProgress(p => ({ ...p, logs: [...p.logs, { type, msg, time: new Date().toLocaleTimeString() }] }));
  };

  const browse = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.style.display = 'none';
    document.body.appendChild(input);
    input.onchange = () => { if (input.files?.[0]?.name) setModelPath(input.files[0].name); };
    input.click();
  };

  const onQuantize = async () => {
    if (!modelPath && !repoId) { alert('Select a model source'); return; }
    setRunning(true);
    setResult(null);
    setError(null);
    setProgress({ percent: 0, text: 'Starting quantization...', logs: [] });
    addLog('info', `Algorithm: ${algorithm}`);
    addLog('info', `Format: ${format}`);
    addLog('info', `Target: ${target}`);

    try {
      const payload: Record<string, unknown> = {
        model_path: modelPath || undefined,
        repo_id: repoId || undefined,
        revision,
        algorithm,
        granularity,
        group_size: groupSize,
        calibrate,
        quantize_embeddings: quantizeEmbeddings,
        quantize_final_norm: quantizeFinalNorm,
        quantize_bias: quantizeBias,
        format,
        export_target: target,
        device,
        layer_exclude_pattern: layerExclude || undefined,
        layer_include_pattern: layerInclude || undefined,
        optimize_for_inference: optimizeInference,
      };
      const init = await quantize(payload);
      if (!init?.job_id) throw new Error('Quantization did not return a job id');
      setJobId(init.job_id);
      addLog('info', `Job started: ${init.job_id}`);
    } catch (e) {
      console.error('[app] quantize error', e);
      setError(e instanceof Error ? e.message : String(e));
      addLog('error', e instanceof Error ? e.message : String(e));
      setRunning(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header style={{ padding: 16, background: '#1e293b', borderBottom: '1px solid #334155' }}>
        <h1 style={{ fontSize: 18, fontWeight: 700, background: 'linear-gradient(135deg,#6366f1,#a855f7)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
          BitForge
        </h1>
      </header>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <Sidebar
          sourceType={sourceType}
          modelPath={modelPath}
          repoId={repoId}
          revision={revision}
          algorithm={algorithm}
          granularity={granularity}
          groupSize={groupSize}
          calibrate={calibrate}
          quantizeEmbeddings={quantizeEmbeddings}
          quantizeFinalNorm={quantizeFinalNorm}
          quantizeBias={quantizeBias}
          optimizeInference={optimizeInference}
          format={format}
          target={target}
          device={device}
          layerInclude={layerInclude}
          layerExclude={layerExclude}
          running={running}
          apiReady={apiReady}
          onQuantize={onQuantize}
          onBrowse={browse}
          currentJobId={jobId ?? ''}
          onCancel={cancelJob}
          onChange={patch => {
            if ('sourceType' in patch) setSourceType(patch.sourceType as SourceType);
            if ('modelPath' in patch) setModelPath(patch.modelPath as string);
            if ('repoId' in patch) setRepoId(patch.repoId as string);
            if ('revision' in patch) setRevision(patch.revision as string);
            if ('algorithm' in patch) setAlgorithm(patch.algorithm as string);
            if ('granularity' in patch) setGranularity(patch.granularity as string);
            if ('groupSize' in patch) setGroupSize(patch.groupSize as number);
            if ('calibrate' in patch) setCalibrate(patch.calibrate as boolean);
            if ('quantizeEmbeddings' in patch) setQuantizeEmbeddings(patch.quantizeEmbeddings as boolean);
            if ('quantizeFinalNorm' in patch) setQuantizeFinalNorm(patch.quantizeFinalNorm as boolean);
            if ('quantizeBias' in patch) setQuantizeBias(patch.quantizeBias as boolean);
            if ('optimizeInference' in patch) setOptimizeInference(patch.optimizeInference as boolean);
            if ('format' in patch) setFormat(patch.format as string);
            if ('target' in patch) setTarget(patch.target as string);
            if ('device' in patch) setDevice(patch.device as string);
            if ('layerInclude' in patch) setLayerInclude(patch.layerInclude as string);
            if ('layerExclude' in patch) setLayerExclude(patch.layerExclude as string);
          }}
        />

        <JobMonitor apiReady={apiReady} progress={progress} result={result} error={error} />
      </div>
    </div>
  );
}

export default App;
