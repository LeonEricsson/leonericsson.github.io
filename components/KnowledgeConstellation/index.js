'use client';

import React, { useMemo, useRef, useState, useCallback, Suspense, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';

// Cluster themes - refined, elegant colors
const CLUSTER_THEMES = {
  0: { name: 'Efficiency & Hardware', color: '#dc2626', glow: '#ff6b6b' },
  1: { name: 'Multimodal & Vision', color: '#7c3aed', glow: '#a78bfa' },
  2: { name: 'Reasoning & RL', color: '#0891b2', glow: '#22d3ee' },
  3: { name: 'Evaluation & Behavior', color: '#059669', glow: '#34d399' },
  4: { name: 'Interpretability', color: '#d97706', glow: '#fbbf24' },
  5: { name: 'Theory & Generalization', color: '#db2777', glow: '#f472b6' },
  6: { name: 'Scaling & Knowledge', color: '#2563eb', glow: '#60a5fa' },
  7: { name: 'Math & Reasoning', color: '#0d9488', glow: '#2dd4bf' },
};

const CLUSTER_ORDER = [0, 1, 2, 3, 4, 5, 6, 7];

// Floating particle background
function AmbientParticles({ count = 200 }) {
  const mesh = useRef();
  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 40;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 40;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 40;
    }
    return pos;
  }, [count]);

  useFrame((state) => {
    if (mesh.current) {
      mesh.current.rotation.y = state.clock.elapsedTime * 0.02;
      mesh.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.01) * 0.1;
    }
  });

  return (
    <points ref={mesh}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.05}
        color="#a0a0a0"
        transparent
        opacity={0.4}
        sizeAttenuation
      />
    </points>
  );
}

// Constellation line between papers
function ConstellationLine({ start, end, opacity = 0.15, color = '#ffffff' }) {
  const ref = useRef();

  const points = useMemo(() => {
    const pts = [];
    const segments = 20;
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      pts.push(new THREE.Vector3(
        start[0] * (1 - t) + end[0] * t,
        start[1] * (1 - t) + end[1] * t,
        start[2] * (1 - t) + end[2] * t
      ));
    }
    return pts;
  }, [start, end]);

  const lineGeometry = useMemo(() => {
    return new THREE.BufferGeometry().setFromPoints(points);
  }, [points]);

  useFrame((state) => {
    if (ref.current) {
      const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.5 + 0.5;
      ref.current.material.opacity = opacity * (0.5 + pulse * 0.3);
    }
  });

  return (
    <line ref={ref} geometry={lineGeometry}>
      <lineBasicMaterial transparent opacity={opacity} color={color} />
    </line>
  );
}

// Individual paper node (glowing orb)
function PaperNode({ data, position, isActive, onHover, onClick, isSelected }) {
  const meshRef = useRef();
  const glowRef = useRef();
  const [hovered, setHovered] = useState(false);
  const theme = CLUSTER_THEMES[data.cluster];

  const targetScale = hovered ? 1.8 : isSelected ? 1.5 : 1;

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.scale.lerp(
        new THREE.Vector3(targetScale, targetScale, targetScale),
        0.1
      );
      meshRef.current.rotation.y += 0.01;
    }
    if (glowRef.current) {
      glowRef.current.scale.lerp(
        new THREE.Vector3(targetScale * 1.5, targetScale * 1.5, targetScale * 1.5),
        0.1
      );
    }
  });

  const handlePointerOver = (e) => {
    e.stopPropagation();
    setHovered(true);
    onHover(data);
  };

  const handlePointerOut = () => {
    setHovered(false);
    onHover(null);
  };

  const handleClick = (e) => {
    e.stopPropagation();
    onClick(data);
  };

  if (!isActive) return null;

  return (
    <group position={position}>
      {/* Glow effect */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.4, 32, 32]} />
        <meshBasicMaterial
          color={theme.glow}
          transparent
          opacity={hovered ? 0.4 : 0.15}
        />
      </mesh>

      {/* Main orb */}
      <mesh
        ref={meshRef}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
        onClick={handleClick}
      >
        <sphereGeometry args={[0.25, 32, 32]} />
        <meshStandardMaterial
          color={theme.color}
          emissive={theme.color}
          emissiveIntensity={hovered ? 1.5 : 0.5}
          metalness={0.3}
          roughness={0.4}
        />
      </mesh>
    </group>
  );
}

// Camera controller for subtle movement
function CameraRig({ isActive }) {
  const { camera } = useThree();
  const targetPos = useRef(new THREE.Vector3(0, 0, 15));

  useFrame((state) => {
    if (isActive) {
      targetPos.current.x = Math.sin(state.clock.elapsedTime * 0.1) * 0.5;
      targetPos.current.y = Math.cos(state.clock.elapsedTime * 0.08) * 0.3;
    }
    camera.position.lerp(targetPos.current, 0.02);
    camera.lookAt(0, 0, 0);
  });

  return null;
}

// Main constellation visualization
function KnowledgeConstellation({ papers, activeClusters, onSelect, selectedPaper, setHoveredWorld }) {
  const { nodes, connections } = useMemo(() => {
    if (papers.length === 0) return { nodes: [], connections: [] };

    // Normalize positions
    const xExtent = { min: Infinity, max: -Infinity };
    const yExtent = { min: Infinity, max: -Infinity };

    papers.forEach(p => {
      if (p.pos) {
        xExtent.min = Math.min(xExtent.min, p.pos[0]);
        xExtent.max = Math.max(xExtent.max, p.pos[0]);
        yExtent.min = Math.min(yExtent.min, p.pos[1]);
        yExtent.max = Math.max(yExtent.max, p.pos[1]);
      }
    });

    const xRange = xExtent.max - xExtent.min || 1;
    const yRange = yExtent.max - yExtent.min || 1;

    // Create nodes
    const nodes = papers.map((paper) => {
      let x = 0, y = 0, z = 0;
      if (paper.pos) {
        x = ((paper.pos[0] - xExtent.min) / xRange - 0.5) * 14;
        y = ((paper.pos[1] - yExtent.min) / yRange - 0.5) * 10;
        z = (Math.random() - 0.5) * 2;
      } else {
        x = (Math.random() - 0.5) * 14;
        y = (Math.random() - 0.5) * 10;
        z = (Math.random() - 0.5) * 2;
      }
      return { ...paper, position: [x, y, z], originalZ: z };
    });

    // Create connections between nearby papers
    const connections = [];
    const connectionThreshold = 3.5;

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const n1 = nodes[i];
        const n2 = nodes[j];
        const dx = n1.position[0] - n2.position[0];
        const dy = n1.position[1] - n2.position[1];
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < connectionThreshold) {
          connections.push({
            start: n1.position,
            end: n2.position,
            opacity: Math.max(0.05, 1 - dist / connectionThreshold) * 0.2
          });
        }
      }
    }

    return { nodes, connections };
  }, [papers]);

  const [hoveredNode, setHoveredNode] = useState(null);

  const handleHover = useCallback((data) => {
    setHoveredNode(data);
    setHoveredWorld(data);
  }, [setHoveredWorld]);

  const handleClick = useCallback((data) => {
    onSelect(data);
  }, [onSelect]);

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, 10]} intensity={0.5} color="#a78bfa" />

      <AmbientParticles count={150} />

      {connections.map((conn, i) => (
        <ConstellationLine key={i} start={conn.start} end={conn.end} opacity={conn.opacity} />
      ))}

      {nodes.map((node, i) => (
        <PaperNode
          key={node.id || i}
          data={node}
          position={node.position}
          isActive={activeClusters.has(node.cluster)}
          onHover={handleHover}
          onClick={handleClick}
          isSelected={selectedPaper?.id === node.id}
        />
      ))}

      <CameraRig isActive={!hoveredNode} />
    </>
  );
}

// Paper detail modal
function PaperModal({ paper, onClose }) {
  if (!paper) return null;

  const theme = CLUSTER_THEMES[paper.cluster];

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <div className="absolute inset-0 bg-black/20 backdrop-blur-sm" />

        <motion.div
          initial={{ opacity: 0, scale: 0.9, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 20 }}
          className="relative max-w-2xl w-full rounded-2xl overflow-hidden shadow-2xl"
          style={{ backgroundColor: 'rgb(255, 254, 249)' }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header with cluster color */}
          <div
            className="px-6 py-5 border-b"
            style={{ borderColor: `${theme.color}30` }}
          >
            <div className="flex items-center justify-between">
              <span
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono text-white"
                style={{ backgroundColor: theme.color }}
              >
                {theme.name}
              </span>
              <button
                onClick={onClose}
                className="p-1 rounded-lg hover:bg-gray-100 transition"
              >
                <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="p-6">
            <h2 className="text-xl font-medium text-gray-900 leading-snug mb-4 font-mono" style={{ fontFamily: 'Courier Prime, monospace' }}>
              {paper.title}
            </h2>

            {paper.summary && (
              <p className="text-sm text-gray-600 leading-relaxed mb-6 font-mono" style={{ fontFamily: 'Courier Prime, monospace' }}>
                {paper.summary}
              </p>
            )}

            {/* Meta */}
            <div className="flex items-center gap-2 text-xs font-mono text-gray-400 mb-6">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
              <a
                href={paper.link}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:underline truncate"
              >
                {paper.link ? new URL(paper.link).hostname : 'Unknown'}
              </a>
            </div>

            {/* Action button */}
            <a
              href={paper.link}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-mono text-white rounded-lg transition-all hover:shadow-lg hover:-translate-y-0.5"
              style={{ backgroundColor: theme.color }}
            >
              <span>Read Paper</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

export default function KnowledgeConstellationWrapper({ mergedData, readingList }) {
  const [activeClusters, setActiveClusters] = useState(new Set(CLUSTER_ORDER));
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [hoveredPaper, setHoveredPaper] = useState(null);

  // Filter papers based on search and clusters
  const filteredPapers = useMemo(() => {
    return mergedData.filter(paper => {
      const matchesCluster = activeClusters.has(paper.cluster);
      const matchesSearch = !searchQuery ||
        paper.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (paper.summary && paper.summary.toLowerCase().includes(searchQuery.toLowerCase()));
      return matchesCluster && matchesSearch;
    });
  }, [mergedData, activeClusters, searchQuery]);

  const toggleCluster = useCallback((clusterId) => {
    setActiveClusters(prev => {
      const next = new Set(prev);
      if (next.has(clusterId)) {
        next.delete(clusterId);
      } else {
        next.add(clusterId);
      }
      return next;
    });
  }, []);

  const selectAllClusters = useCallback(() => setActiveClusters(new Set(CLUSTER_ORDER)), []);
  const clearAllClusters = useCallback(() => setActiveClusters(new Set()), []);

  return (
    <div className="relative w-full" style={{ height: 'calc(100vh - 140px)', minHeight: '500px' }}>
      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 0, 15], fov: 50 }}
        style={{ background: 'linear-gradient(180deg, rgb(255, 254, 249) 0%, rgb(250, 249, 245) 100%)' }}
      >
        <Suspense fallback={null}>
          <KnowledgeConstellation
            papers={filteredPapers}
            activeClusters={activeClusters}
            onSelect={setSelectedPaper}
            selectedPaper={selectedPaper}
            setHoveredWorld={setHoveredPaper}
          />
        </Suspense>
      </Canvas>

      {/* Floating UI Controls */}
      <div className="absolute top-0 left-0 right-0 p-4 pointer-events-none">
        <div className="max-w-7xl mx-auto pointer-events-auto">
          {/* Header */}
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-4">
            <div>
              <h1 className="text-2xl font-light tracking-tight text-gray-900 font-mono">
                Reading Archive
              </h1>
              <p className="text-sm text-gray-500 mt-1 font-mono">
                {filteredPapers.length} papers across {activeClusters.size} topics
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              {/* Search */}
              <div className="relative group">
                <input
                  type="text"
                  placeholder="Search papers..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full lg:w-56 px-4 py-2 pl-9 bg-white/90 backdrop-blur-sm border border-gray-200 rounded-lg
                           focus:outline-none focus:ring-2 focus:ring-gray-200 focus:border-gray-400
                           text-sm transition-all font-mono placeholder-gray-400"
                />
                <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 group-focus-within:text-gray-600 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>

              {/* Quick filters */}
              <div className="flex gap-1">
                <button onClick={selectAllClusters} className="px-2 py-1.5 text-xs font-mono bg-white/90 backdrop-blur-sm border border-gray-200 text-gray-600 rounded hover:bg-gray-50 transition">
                  All
                </button>
                <button onClick={clearAllClusters} className="px-2 py-1.5 text-xs font-mono bg-white/90 backdrop-blur-sm border border-gray-200 text-gray-600 rounded hover:bg-gray-50 transition">
                  None
                </button>
              </div>
            </div>
          </div>

          {/* Cluster filters */}
          <div className="flex flex-wrap gap-2">
            {CLUSTER_ORDER.map(clusterId => {
              const theme = CLUSTER_THEMES[clusterId];
              const isActive = activeClusters.has(clusterId);
              const count = mergedData.filter(n => n.cluster === clusterId).length;

              return (
                <button
                  key={clusterId}
                  onClick={() => toggleCluster(clusterId)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-mono
                             border transition-all duration-200 cursor-pointer ${
                              isActive
                                ? 'border-transparent text-white shadow-sm'
                                : 'border-gray-200 bg-white/90 backdrop-blur-sm text-gray-500 hover:border-gray-300'
                            }`}
                  style={{ backgroundColor: isActive ? theme.color : undefined }}
                >
                  <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-white' : ''}`} style={{ backgroundColor: isActive ? undefined : theme.color }} />
                  <span>{theme.name.split(' & ')[0]}</span>
                  <span className={isActive ? 'text-white/70' : 'text-gray-400'}>({count})</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Instructions */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 pointer-events-none">
        <div className="flex items-center gap-6 px-4 py-2 rounded-full text-xs font-mono bg-white/80 backdrop-blur-sm shadow-sm">
          <span className="flex items-center gap-1.5 text-gray-500">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5" /></svg>
            Hover to explore
          </span>
          <span className="flex items-center gap-1.5 text-gray-500">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 15l-2 5" /></svg>
            Click to select
          </span>
        </div>
      </div>

      {/* Stats */}
      <div className="absolute top-4 right-4 pointer-events-none">
        <div className="px-4 py-2 rounded-xl bg-white/90 backdrop-blur-sm shadow-sm">
          <p className="text-2xl font-light text-gray-900 font-mono">{filteredPapers.length}</p>
          <p className="text-xs text-gray-500 font-mono">Resources</p>
        </div>
      </div>

      {/* Hovered paper indicator */}
      {hoveredPaper && (
        <div className="absolute top-4 left-4 px-4 py-2 rounded-xl bg-white/90 backdrop-blur-sm shadow-sm animate-fade-in">
          <span
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono text-white"
            style={{ backgroundColor: CLUSTER_THEMES[hoveredPaper.cluster].color }}
          >
            {CLUSTER_THEMES[hoveredPaper.cluster].name}
          </span>
        </div>
      )}

      {/* Paper modal */}
      <PaperModal paper={selectedPaper} onClose={() => setSelectedPaper(null)} />

      {/* Empty state */}
      {filteredPapers.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
              <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-gray-500 font-mono text-sm mb-4">No papers match your filters</p>
          </div>
        </div>
      )}

      <style jsx global>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(-10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fade-in 0.2s ease-out;
        }
      `}</style>
    </div>
  );
}
