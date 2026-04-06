import React from 'react';
import { User, Activity, Database } from 'lucide-react';
import type { SliceData } from '../App';

interface SidebarLeftProps {
  currentSlice: number;
  setSlice: (id: number) => void;
  slices: SliceData[];
  totalSlices: number;
}

function getStatus(score: number): 'normal' | 'suspicious' | 'anomaly' {
  if (score > 0.65) return 'anomaly';
  if (score > 0.45) return 'suspicious';
  return 'normal';
}

export default function SidebarLeft({ currentSlice, setSlice, slices, totalSlices }: SidebarLeftProps) {
  // Fallback mock slices if no real data yet
  const displaySlices = slices.length > 0
    ? slices
    : Array.from({ length: totalSlices }, (_, i) => ({
        id: i, base: '', score: i > 20 && i < 30 ? 0.7 : 0.2,
        label: 0, subject: 0, has_gt: false, region: 'mid-superior'
      }));

  const currentSliceData = displaySlices[currentSlice];

  return (
    <div className="sidebar" style={{ width: '280px' }}>
      <div className="panel animate-slide-down">
        <h2 className="panel-title">Patient Overview</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <User size={14} /> Subject ID
            </span>
            <span className="metric-value" style={{ fontSize: '0.9rem' }}>
              {currentSliceData ? `BraTS-S${currentSliceData.subject.toString().padStart(3,'0')}` : 'IXI-0012-T1'}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={14} /> Modality
            </span>
            <span style={{ color: 'var(--text-primary)', fontSize: '0.9rem' }}>T1-weighted</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Database size={14} /> Region
            </span>
            <span style={{ color: 'var(--text-primary)', fontSize: '0.9rem' }}>
              {currentSliceData?.region ?? 'mid-superior'}
            </span>
          </div>
          {currentSliceData && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Anomaly Score</span>
              <span className="metric-value" style={{
                fontSize: '1rem',
                color: currentSliceData.score > 0.65 ? '#e24b4a' :
                       currentSliceData.score > 0.45 ? '#f0a500' : '#1d9e75'
              }}>
                {(currentSliceData.score * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0 }}>
        <div style={{ padding: '1rem' }}>
          <h2 className="panel-title">Confidence Filmstrip</h2>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
            Axial slices. Colors indicate anomaly probability.
          </p>
        </div>
        <div className="ribbon-section" style={{ overflowY: 'auto', flex: 1, padding: '0 1rem 1rem 1rem' }}>
          {displaySlices.map((slice, idx) => {
            const status = getStatus(slice.score);
            return (
              <div
                key={idx}
                className={`thumbnail ${idx === currentSlice ? 'active' : ''}`}
                onClick={() => setSlice(idx)}
                title={`Slice ${idx} — score: ${(slice.score * 100).toFixed(1)}%`}
              >
                {slice.base ? (
                  <img
                    src={`/images/${slice.base}_original.png`}
                    alt={`slice ${idx}`}
                    style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', borderRadius: '4px' }}
                  />
                ) : (
                  <div style={{
                    position: 'absolute', inset: '10%', borderRadius: '50%',
                    background: 'radial-gradient(circle, rgba(200,200,200,0.6) 0%, rgba(50,50,50,0.2) 70%, transparent 100%)',
                    opacity: 0.5
                  }} />
                )}
                <span style={{ position: 'absolute', top: 2, left: 4, fontSize: '0.6rem', color: '#fff', zIndex: 10, fontFamily: 'var(--font-mono)' }}>
                  {idx}
                </span>
                <div className={`ribbon ${status}`} />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
