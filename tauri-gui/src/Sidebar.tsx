import { useState } from 'react';

type SourceType = 'path' | 'hub';

const API_BASE = `http://${window.location.hostname}:8125`;

interface SidebarProps {
  sourceType: SourceType;
  modelPath: string;
  repoId: string;
  revision: string;
  algorithm: string;
  granularity: string;
  groupSize: number;
  calibrate: boolean;
  quantizeEmbeddings: boolean;
  quantizeFinalNorm: boolean;
  quantizeBias: boolean;
  optimizeInference: boolean;
  format: string;
  target: string;
  device: string;
  layerInclude: string;
  layerExclude: string;
  running: boolean;
  apiReady: boolean;
  currentJobId: string;
  onQuantize: () => void;
  onBrowse: () => void;
  onCancel: () => void;
  onChange: (patch: Record<string, any>) => void;
}

export function Sidebar({
  sourceType, modelPath, repoId, revision, algorithm, granularity, groupSize,
  calibrate, quantizeEmbeddings, quantizeFinalNorm, quantizeBias, optimizeInference,
  format, target, device, layerInclude, layerExclude, running, apiReady, currentJobId, onQuantize, onBrowse, onCancel, onChange
}: SidebarProps) {
  const input = (name: string, props: any = {}) => (
    <input
      name={name}
      value={(props.value as any) ?? ''}
      onChange={e => onChange({ [name]: props.type === 'number' ? Number(e.target.value) : e.target.value })}
      {...props}
      style={{ width: '100%', padding: 8, borderRadius: 6, border: '1px solid #334155', background: '#0f172a', color: '#e2e8f0', marginBottom: 8 }}
    />
  );

  return (
    <aside style={{ width: 320, background: '#0f172a', borderRight: '1px solid #334155', padding: 16, overflowY: 'auto' }}>
      <section style={{ marginBottom: 16, padding: 12, background: '#1e293b', borderRadius: 12, border: '1px solid #334155' }}>
        <div style={sectionTitle}>Model</div>
        <label style={label}>Source</label>
        <select value={sourceType} onChange={e => onChange({ sourceType: e.target.value })} style={select}>
          <option value="path">Local Path</option>
          <option value="hub">Hugging Face Hub</option>
        </select>
        {sourceType === 'path' ? (
          <>
            {input('modelPath', { value: modelPath, placeholder: 'C:\\Models\\model.pt' })}
            <button onClick={onBrowse} style={secondaryButton}>Browse</button>
          </>
        ) : (
          <>
            {input('repoId', { value: repoId, placeholder: 'org/model or https://huggingface.co/org/model' })}
            {input('revision', { value: revision, placeholder: 'main' })}
          </>
        )}
        {sourceType === 'hub' && repoId && apiReady && (
          <button onClick={async () => {
            try {
              const info = await fetch(`${API_BASE}/api/hub/info?model=${encodeURIComponent(repoId)}&revision=${encodeURIComponent(revision)}`).then(r => r.json());
              alert(`Repo: ${info.model ?? repoId}\nRevision: ${info.revision ?? 'main'}\nTags: ${(info.tags ?? []).slice(0, 5).join(', ') || '-'}`);
            } catch (e) {
              alert((e instanceof Error ? e.message : String(e)) || 'Model lookup failed');
            }
          }} style={{ ...secondaryButton, marginTop: 8 }}>Lookup Model</button>
        )}
      </section>

      <section style={{ marginBottom: 16, padding: 12, background: '#1e293b', borderRadius: 12, border: '1px solid #334155' }}>
        <div style={sectionTitle}>Quantization</div>
        <label style={label}>Algorithm</label>
        <select value={algorithm} onChange={e => onChange({ algorithm: e.target.value })} style={select}>
          <option value="xnor">XNOR</option>
          <option value="binarize">BinaryNet / Binarize</option>
          <option value="irnet">IRNet</option>
          <option value="adaptive">Adaptive</option>
        </select>

        <label style={label}>Granularity</label>
        <select value={granularity} onChange={e => onChange({ granularity: e.target.value })} style={select}>
          <option value="per_channel">Per Channel</option>
          <option value="per_tensor">Per Tensor</option>
          <option value="per_group">Per Group</option>
        </select>

        <label style={label}>Group Size</label>
        {input('groupSize', { type: 'number', value: groupSize, min: 8, max: 128, step: 8 })}

        <Checkbox name="calibrate" label="Calibration" checked={calibrate} onChange={onChange} />
        <Checkbox name="quantizeEmbeddings" label="Embeddings" checked={quantizeEmbeddings} onChange={onChange} />
        <Checkbox name="quantizeFinalNorm" label="Final Norm" checked={quantizeFinalNorm} onChange={onChange} />
        <Checkbox name="quantizeBias" label="Bias" checked={quantizeBias} onChange={onChange} />
        <Checkbox name="optimizeInference" label="Optimize Inference" checked={optimizeInference} onChange={onChange} />
      </section>

      <section style={{ marginBottom: 16, padding: 12, background: '#1e293b', borderRadius: 12, border: '1px solid #334155' }}>
        <div style={sectionTitle}>Export</div>
        <label style={label}>Format</label>
        <select value={format} onChange={e => onChange({ format: e.target.value })} style={select}>
          <option value="gguf">GGUF</option>
          <option value="safetensors">SafeTensors</option>
          <option value="pytorch">PyTorch</option>
          <option value="onnx">ONNX</option>
          <option value="torchscript">TorchScript</option>
        </select>

        <label style={label}>Target</label>
        <select value={target} onChange={e => onChange({ target: e.target.value })} style={select}>
          <option value="mobile">Mobile</option>
          <option value="desktop">Desktop</option>
          <option value="edge">Edge</option>
        </select>

        <label style={label}>Device</label>
        <select value={device} onChange={e => onChange({ device: e.target.value })} style={select}>
          <option value="auto">Auto</option>
          <option value="cpu">CPU</option>
          <option value="cuda">CUDA</option>
          <option value="metal">Metal</option>
        </select>
      </section>

      <section style={{ marginBottom: 16, padding: 12, background: '#1e293b', borderRadius: 12, border: '1px solid #334155' }}>
        <div style={sectionTitle}>Advanced</div>
        <label style={label}>Layer Include</label>
        {input('layerInclude', { value: layerInclude, placeholder: 'encoder.layer' })}
        <label style={label}>Layer Exclude</label>
        {input('layerExclude', { value: layerExclude, placeholder: 'lm_head, embed' })}
      </section>

      <button onClick={onQuantize} disabled={running || !apiReady} style={{
        width:'100%',padding:14,borderRadius:8,border:'none',
        background:running||!apiReady?'#475569':'linear-gradient(135deg,#6366f1,#a855f7)',
        color:'white',fontWeight:700,cursor:running||!apiReady?'not-allowed':'pointer',
        opacity:running||!apiReady?.7:1
      }}>
        {running ? '⏳ Quantizing...' : '⚡ Quantize Model'}
      </button>
      {running && (
        <button onClick={onCancel} style={{
          width:'100%',marginTop:10,padding:10,borderRadius:8,border:'1px solid #f87171',
          background:'transparent',color:'#f87171',fontWeight:600,cursor:'pointer'
        }}>
          Cancel
        </button>
      )}
    </aside>
  );
}

function Checkbox({ name, label, checked, onChange }: { name: string; label: string; checked: boolean; onChange: (p: Record<string, any>) => void }) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, fontSize: 13, color: '#cbd5e1' }}>
      <input type="checkbox" checked={checked} onChange={e => onChange({ [name]: e.target.checked })} />
      {label}
    </label>
  );
}

const sectionTitle: Record<string, any> = { fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#94a3b8', marginBottom: 8 };
const label: Record<string, any> = { fontSize: 12, color: '#cbd5e1', marginBottom: 4, display: 'block' };
const select: Record<string, any> = { width: '100%', padding: 8, borderRadius: 6, border: '1px solid #334155', background: '#0f172a', color: '#e2e8f0', marginBottom: 8 };
const secondaryButton: Record<string, any> = { width: '100%', padding: 8, borderRadius: 6, border: '1px solid #334155', background: '#334155', color: '#e2e8f0', cursor: 'pointer' };
