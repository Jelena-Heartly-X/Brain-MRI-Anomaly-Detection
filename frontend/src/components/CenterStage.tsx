import React, { useState } from 'react';
import { Maximize2, Layers, Box } from 'lucide-react';
import type { SliceData } from '../App';

interface CenterStageProps {
  currentSlice: number;
  setSlice: (val: number) => void;
  slices: SliceData[];
  totalSlices: number;
}

export default function CenterStage({ currentSlice, setSlice, slices, totalSlices }: CenterStageProps) {
  const [mode, setMode] = useState<'slice' | 'volume'>('slice');
  const [opacity, setOpacity] = useState(0.6);

  const slice = slices[currentSlice];
  const base = slice?.base;

  // Panels: label → image suffix
  const panels: { label: string; suffix: string; fallbackStyle: React.CSSProperties }[] = [
    {
      label: 'Original Input',
      suffix: '_original.png',
      fallbackStyle: { background: 'radial-gradient(circle, #ddd 0%, #333 70%, #000 100%)' }
    },
    {
      label: 'Masked Input',
      suffix: '_masked.png',
      fallbackStyle: { background: 'radial-gradient(circle, #333 0%, #111 70%, #000 100%)' }
    },
    {
      label: 'Reconstruction',
      suffix: '_recon.png',
      fallbackStyle: { background: 'radial-gradient(circle, #ddd 0%, #333 70%, #000 100%)' }
    },
    {
      label: 'Error Map',
      suffix: '_error.png',
      fallbackStyle: { background: `radial-gradient(circle at 60% 40%, rgba(226,75,74,${opacity}) 0%, rgba(186,117,23,${opacity*0.5}) 30%, transparent 60%)` }
    },
    {
      label: 'Heatmap Overlay',
      suffix: '_heatmap.png',
      fallbackStyle: { background: `radial-gradient(circle at 60% 40%, rgba(226,75,74,${opacity}) 0%, rgba(186,117,23,${opacity*0.5}) 30%, transparent 60%)` }
    },
    {
      label: 'Predicted Mask',
      suffix: '_error.png', // use error map as proxy
      fallbackStyle: { background: 'radial-gradient(circle at 60% 40%, #fff 0%, #fff 20%, transparent 22%)' }
    },
  ];

  const isAnomalous = (slice?.score ?? 0) > 0.5;

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-main)' }}>
      {/* Top Bar */}
      <div className="tabs">
        <div className={`tab ${mode === 'slice' ? 'active' : ''}`} onClick={() => setMode('slice')}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={16} /> 2D Slice Mode
        </div>
        <div className={`tab ${mode === 'volume' ? 'active' : ''}`} onClick={() => setMode('volume')}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Box size={16} /> 3D Volume Mode
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', paddingRight: '1rem' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Heatmap Blend</span>
          <input type="range" min="0" max="1" step="0.05" value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            style={{ width: '100px', accentColor: 'var(--color-accent)' }} />
        </div>
      </div>

      {mode === 'slice' ? (
        <div className="viewer-grid" style={{ flex: 1 }}>
          {panels.map(({ label, suffix, fallbackStyle }, i) => (
            <div key={label} className="viewer-panel">
              <div className="viewer-label">{label}</div>
              {base ? (
                <img
                  src={`/images/${base}${suffix}`}
                  alt={label}
                  style={{
                    width: '85%', height: '85%', objectFit: 'contain',
                    opacity: i === 4 ? opacity : 1,
                    filter: i === 3 ? 'drop-shadow(0 0 8px rgba(226,75,74,0.6))' : 'none',
                    borderRadius: '4px'
                  }}
                  className={i > 2 && isAnomalous ? 'pulse-anomaly' : ''}
                />
              ) : (
                <div style={{
                  width: '60%', height: '80%', borderRadius: '40% 40% 50% 50%',
                  ...fallbackStyle,
                  filter: i === 3 ? 'drop-shadow(0 0 10px red)' : 'none',
                  opacity: i === 4 ? opacity : 1,
                  border: i === 5 ? '1px solid #e24b4a' : 'none'
                }} className={i > 2 && isAnomalous ? 'pulse-anomaly' : ''} />
              )}
              <button style={{ position: 'absolute', bottom: '0.5rem', right: '0.5rem', background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                <Maximize2 size={16} />
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
          <div className="viewer-label">3D Brain Volume</div>
          {base ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', maxWidth: '600px' }}>
              {['_original.png', '_recon.png', '_error.png', '_heatmap.png'].map((s, i) => (
                <img key={i} src={`/images/${base}${s}`} alt={s}
                  style={{ width: '100%', borderRadius: '8px', border: '1px solid var(--border)' }} />
              ))}
            </div>
          ) : (
            <div style={{ width: '400px', height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <p style={{ color: 'var(--text-secondary)' }}>3D view available with real scan data</p>
            </div>
          )}
          <div style={{ position: 'absolute', top: '40%', left: '10%', right: '10%', height: '2px', background: 'var(--color-accent)', boxShadow: '0 0 10px var(--color-accent)', transform: `translateY(${(currentSlice - (totalSlices/2)) * 3}px)` }} />
        </div>
      )}

      {/* Scrubber */}
      <div className="scrubber-container">
        <span className="metric-value" style={{ fontSize: '1rem' }}>S{currentSlice.toString().padStart(3, '0')}</span>
        <input type="range" min="0" max={totalSlices - 1} value={currentSlice}
          onChange={(e) => setSlice(parseInt(e.target.value))}
          className="scrubber-slider" />
        <span className="metric-value" style={{ fontSize: '1rem' }}>S{(totalSlices - 1).toString().padStart(3, '0')}</span>
      </div>
    </div>
  );
}
