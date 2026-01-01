'use client';

import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Pastel cluster themes - balanced clarity and brightness
const CLUSTER_THEMES = {
  0: { color: '#f5a0af', soft: 'rgba(245, 160, 175, 0.08)', accent: '#f5a0af' }, // blush pink
  1: { color: '#b6a0fb', soft: 'rgba(182, 160, 251, 0.08)', accent: '#b6a0fb' }, // soft lavender
  2: { color: '#91d6fe', soft: 'rgba(145, 214, 254, 0.08)', accent: '#91d6fe' }, // sky blue
  3: { color: '#9cf1b4', soft: 'rgba(156, 241, 180, 0.08)', accent: '#9cf1b4' }, // mint
  4: { color: '#fec98e', soft: 'rgba(254, 201, 142, 0.08)', accent: '#fec98e' }, // apricot
  5: { color: '#c8a2fe', soft: 'rgba(200, 162, 254, 0.08)', accent: '#c8a2fe' }, // lilac
  6: { color: '#80e9f6', soft: 'rgba(128, 233, 246, 0.08)', accent: '#80e9f6' }, // aqua
  7: { color: '#fbb5d6', soft: 'rgba(251, 181, 214, 0.08)', accent: '#fbb5d6' }, // rose
};

// Draw a flat star shape
const drawStar = (ctx, cx, cy, outerRadius, innerRadius, points) => {
  const step = Math.PI / points;
  ctx.beginPath();
  for (let i = 0; i < 2 * points; i++) {
    const radius = i % 2 === 0 ? outerRadius : innerRadius;
    const angle = i * step - Math.PI / 2;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.closePath();
};

const CLUSTER_ORDER = [0, 1, 2, 3, 4, 5, 6, 7];

// Utility to lerp values
const lerp = (a, b, t) => a + (b - a) * t;

// Generate floating particles
const generateParticles = (count, width, height) => {
  return Array.from({ length: count }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    radius: 0.5 + Math.random() * 1.5,
    opacity: 0.1 + Math.random() * 0.2,
    phase: Math.random() * Math.PI * 2,
  }));
};

export default function KnowledgeNebula({ papers = [] }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const animationRef = useRef(null);
  const mouseRef = useRef({ x: 0, y: 0, targetX: 0, targetY: 0 });
  const transformRef = useRef({ x: 0, y: 0, scale: 1, targetScale: 1, targetX: 0, targetY: 0 });
  const nodesRef = useRef([]);
  const connectionsRef = useRef([]);
  const particlesRef = useRef([]);
  const isDraggingRef = useRef(false);
  const dragStartRef = useRef({ x: 0, y: 0 });

  const [activeClusters, setActiveClusters] = useState(new Set(CLUSTER_ORDER));
  const [searchQuery, setSearchQuery] = useState('');
  const [hoveredPaper, setHoveredPaper] = useState(null);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [isInitialized, setIsInitialized] = useState(false);

  // Process and normalize paper positions
  const processedPapers = useMemo(() => {
    if (!papers || papers.length === 0) return [];

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    papers.forEach(p => {
      if (p.pos) {
        minX = Math.min(minX, p.pos[0]);
        maxX = Math.max(maxX, p.pos[0]);
        minY = Math.min(minY, p.pos[1]);
        maxY = Math.max(maxY, p.pos[1]);
      }
    });

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const padding = 0.1;

    return papers.map(paper => {
      let nx = 0.5, ny = 0.5;
      if (paper.pos) {
        nx = padding + ((paper.pos[0] - minX) / rangeX) * (1 - 2 * padding);
        ny = padding + ((paper.pos[1] - minY) / rangeY) * (1 - 2 * padding);
      }
      return {
        ...paper,
        nx, ny,
        vx: 0, vy: 0,
        baseRadius: 4 + Math.random() * 2,
        pulsePhase: Math.random() * Math.PI * 2,
        pulseSpeed: 0.5 + Math.random() * 0.5,
      };
    });
  }, [papers]);

  // Filter papers based on search and active clusters
  const filteredPapers = useMemo(() => {
    return processedPapers.filter(paper => {
      const matchesCluster = activeClusters.has(paper.cluster);
      const matchesSearch = !searchQuery ||
        paper.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        paper.summary?.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesCluster && matchesSearch;
    });
  }, [processedPapers, activeClusters, searchQuery]);

  // Initialize nodes and connections
  useEffect(() => {
    if (filteredPapers.length === 0 || dimensions.width === 0) return;

    const width = dimensions.width;
    const height = dimensions.height;

    // Create node objects with screen positions
    const nodes = filteredPapers.map(paper => ({
      ...paper,
      x: paper.nx * width,
      y: paper.ny * height,
      targetX: paper.nx * width,
      targetY: paper.ny * height,
      radius: paper.baseRadius,
      targetRadius: paper.baseRadius,
      opacity: 1,
      targetOpacity: 1,
    }));

    // Create connections between nearby papers in same cluster
    const connections = [];
    const connectionThreshold = Math.min(width, height) * 0.12;

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const n1 = nodes[i];
        const n2 = nodes[j];
        if (n1.cluster !== n2.cluster) continue;

        const dx = n1.x - n2.x;
        const dy = n1.y - n2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < connectionThreshold) {
          const strength = Math.pow(1 - dist / connectionThreshold, 2);
          connections.push({
            from: i,
            to: j,
            strength,
            opacity: strength * 0.3,
            targetOpacity: strength * 0.3,
          });
        }
      }
    }

    nodesRef.current = nodes;
    connectionsRef.current = connections;
    setIsInitialized(true);
  }, [filteredPapers, dimensions]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Mouse movement handler
  const handleMouseMove = useCallback((e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    mouseRef.current.targetX = x;
    mouseRef.current.targetY = y;

    // Handle dragging
    if (isDraggingRef.current) {
      const dx = x - dragStartRef.current.x;
      const dy = y - dragStartRef.current.y;
      transformRef.current.targetX = transformRef.current.x + dx;
      transformRef.current.targetY = transformRef.current.y + dy;
      transformRef.current.x = transformRef.current.targetX;
      transformRef.current.y = transformRef.current.targetY;
      dragStartRef.current = { x, y };
      return;
    }

    // Check for hover
    const nodes = nodesRef.current;
    let found = null;
    const scale = transformRef.current.scale;

    for (let i = nodes.length - 1; i >= 0; i--) {
      const node = nodes[i];
      const dx = x - (node.x * scale + transformRef.current.x);
      const dy = y - (node.y * scale + transformRef.current.y);
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < (node.radius + 8) * scale) {
        found = node;
        break;
      }
    }

    setHoveredPaper(found);
  }, []);

  // Mouse down handler
  const handleMouseDown = useCallback((e) => {
    if (!hoveredPaper) {
      isDraggingRef.current = true;
      const rect = containerRef.current.getBoundingClientRect();
      dragStartRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }
  }, [hoveredPaper]);

  // Mouse up handler
  const handleMouseUp = useCallback(() => {
    isDraggingRef.current = false;
  }, []);

  // Click handler
  const handleClick = useCallback((e) => {
    if (!isDraggingRef.current && hoveredPaper) {
      setSelectedPaper(hoveredPaper);
    } else if (!isDraggingRef.current) {
      setSelectedPaper(null);
    }
  }, [hoveredPaper]);

  // Wheel handler for zoom (centered on mouse)
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const delta = e.deltaY * -0.01;
    const oldScale = transformRef.current.scale;
    const newScale = Math.max(0.5, Math.min(3, oldScale + delta));

    // Calculate zoom point
    const zoomPointX = (mouseX - transformRef.current.x) / oldScale;
    const zoomPointY = (mouseY - transformRef.current.y) / oldScale;

    // Adjust pan to zoom towards mouse
    transformRef.current.targetX = mouseX - zoomPointX * newScale;
    transformRef.current.targetY = mouseY - zoomPointY * newScale;
    transformRef.current.targetScale = newScale;
  }, []);

  // Reset camera handler
  const resetCamera = useCallback(() => {
    transformRef.current.targetX = 0;
    transformRef.current.targetY = 0;
    transformRef.current.targetScale = 1;
  }, []);

  // Keyboard handler for space to reset
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        resetCamera();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [resetCamera]);

  // Animation loop
  useEffect(() => {
    if (!canvasRef.current || dimensions.width === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = dimensions.width * dpr;
    canvas.height = dimensions.height * dpr;
    ctx.scale(dpr, dpr);

    // Initialize particles if not done
    if (particlesRef.current.length === 0) {
      particlesRef.current = generateParticles(80, dimensions.width, dimensions.height);
    }

    let time = 0;

    const animate = () => {
      time += 0.016;

      const width = dimensions.width;
      const height = dimensions.height;

      // Smooth mouse following
      mouseRef.current.x = lerp(mouseRef.current.x, mouseRef.current.targetX, 0.1);
      mouseRef.current.y = lerp(mouseRef.current.y, mouseRef.current.targetY, 0.1);

      // Smooth scale and pan
      transformRef.current.scale = lerp(transformRef.current.scale, transformRef.current.targetScale, 0.08);
      transformRef.current.x = lerp(transformRef.current.x, transformRef.current.targetX, 0.08);
      transformRef.current.y = lerp(transformRef.current.y, transformRef.current.targetY, 0.08);

      // Calculate parallax offset based on mouse position (subtle when not dragging)
      const parallaxStrength = isDraggingRef.current ? 0 : 15;
      const px = (mouseRef.current.x / width - 0.5) * parallaxStrength;
      const py = (mouseRef.current.y / height - 0.5) * parallaxStrength;

      // Clear canvas with warm background
      ctx.fillStyle = 'rgb(255, 254, 249)';
      ctx.fillRect(0, 0, width, height);

      // Draw floating particles
      const particles = particlesRef.current;
      particles.forEach(p => {
        // Update position with subtle drift
        p.x += p.vx + Math.sin(time * 0.5 + p.phase) * 0.1;
        p.y += p.vy + Math.cos(time * 0.3 + p.phase) * 0.1;

        // Wrap around edges
        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;

        // Draw particle with subtle pulse
        const pulse = Math.sin(time * 2 + p.phase) * 0.3 + 1;
        ctx.globalAlpha = p.opacity * pulse * 0.6;
        ctx.fillStyle = '#d1d5db';
        ctx.beginPath();
        ctx.arc(p.x + px * 0.3, p.y + py * 0.3, p.radius, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;

      const scale = transformRef.current.scale;
      const nodes = nodesRef.current;
      const connections = connectionsRef.current;

      // Draw connections with organic curves
      ctx.lineCap = 'round';
      connections.forEach(conn => {
        const n1 = nodes[conn.from];
        const n2 = nodes[conn.to];
        if (!n1 || !n2) return;

        const x1 = n1.x * scale + transformRef.current.x + px;
        const y1 = n1.y * scale + transformRef.current.y + py;
        const x2 = n2.x * scale + transformRef.current.x + px;
        const y2 = n2.y * scale + transformRef.current.y + py;

        // Animate opacity
        conn.opacity = lerp(conn.opacity, conn.targetOpacity, 0.05);

        const theme = CLUSTER_THEMES[n1.cluster];
        ctx.strokeStyle = theme.color;
        ctx.globalAlpha = conn.opacity * 1.5;
        ctx.lineWidth = 2;

        // Draw curved line
        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2;
        const offset = Math.sin(time * 0.5 + conn.from) * 5;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.quadraticCurveTo(midX + offset, midY + offset, x2, y2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      });

      // Draw nodes as flat pastel stars
      nodes.forEach((node, i) => {
        const theme = CLUSTER_THEMES[node.cluster];
        const isHovered = hoveredPaper?.id === node.id;
        const isSelected = selectedPaper?.id === node.id;

        // Animate radius
        const targetRadius = isHovered || isSelected ? node.baseRadius * 2.2 : node.baseRadius;
        node.radius = lerp(node.radius, targetRadius, 0.15);

        const x = node.x * scale + transformRef.current.x + px;
        const y = node.y * scale + transformRef.current.y + py;
        const r = node.radius * scale;

        // Subtle rotation animation
        const rotation = time * 0.2 + node.pulsePhase;

        // Star dimensions
        const outerR = r * 1.4;
        const innerR = r * 0.55;
        const points = 4; // 4-pointed star for clean look

        // Save context for rotation
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(rotation);

        // Draw flat star - single solid color
        ctx.fillStyle = theme.color;
        ctx.globalAlpha = 1;
        drawStar(ctx, 0, 0, outerR, innerR, points);
        ctx.fill();

        ctx.globalAlpha = 1;
        ctx.restore();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [dimensions, isInitialized, hoveredPaper, selectedPaper]);

  return (
    <div className="relative w-full h-full" ref={containerRef}>
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ cursor: isDraggingRef.current ? 'grabbing' : hoveredPaper ? 'pointer' : 'grab', touchAction: 'none' }}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => { setHoveredPaper(null); isDraggingRef.current = false; }}
        onClick={handleClick}
        onWheel={handleWheel}
      />

      {/* Top Controls */}
      <div className="absolute top-6 right-6 pointer-events-none z-10">
        {/* Search Bar - minimal design */}
        <div className="pointer-events-auto">
          <input
            type="text"
            placeholder="search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-48 px-3 py-1.5 bg-transparent text-right
                       focus:outline-none
                       text-sm transition-all placeholder-gray-400/60"
            style={{ fontFamily: 'Courier Prime, monospace' }}
          />
        </div>
      </div>

      {/* Hover Card */}
      <AnimatePresence>
        {hoveredPaper && !selectedPaper && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            transition={{ duration: 0.15 }}
            className="absolute bottom-6 left-6 max-w-md pointer-events-none z-20"
          >
            <div className="bg-white/95 backdrop-blur-md rounded-xl shadow-lg border border-gray-100 p-4 overflow-hidden">
              <div className="flex items-start gap-3">
                <div
                  className="w-2 h-2 rounded-full flex-shrink-0 mt-1.5"
                  style={{ backgroundColor: CLUSTER_THEMES[hoveredPaper.cluster].color }}
                />
                <div className="flex-1">
                  <h3
                    className="text-sm font-medium text-gray-900 leading-snug mb-1"
                    style={{ fontFamily: 'Courier Prime, monospace' }}
                  >
                    {hoveredPaper.title}
                  </h3>
                  <p className="text-xs text-gray-400" style={{ fontFamily: 'Courier Prime, monospace' }}>
                    Click to view
                  </p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Selected Paper Modal */}
      <AnimatePresence>
        {selectedPaper && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center p-4"
            onClick={() => setSelectedPaper(null)}
          >
            <div className="absolute inset-0 bg-black/10 backdrop-blur-sm" />

            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="relative max-w-2xl w-full bg-white rounded-2xl shadow-2xl overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div
                className="px-6 py-4 border-b flex items-center justify-between"
                style={{ borderColor: CLUSTER_THEMES[selectedPaper.cluster].color + '20' }}
              >
                <div
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: CLUSTER_THEMES[selectedPaper.cluster].color }}
                />
                <button
                  onClick={() => setSelectedPaper(null)}
                  className="p-1.5 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              {/* Content */}
              <div className="p-6">
                <h2
                  className="text-xl font-medium text-gray-900 leading-snug mb-4"
                  style={{ fontFamily: 'Courier Prime, monospace' }}
                >
                  {selectedPaper.title}
                </h2>

                {selectedPaper.summary && (
                  <p
                    className="text-sm text-gray-600 leading-relaxed mb-6 max-h-48 overflow-y-auto"
                    style={{ fontFamily: 'Courier Prime, monospace' }}
                  >
                    {selectedPaper.summary.slice(0, 600)}
                    {selectedPaper.summary.length > 600 && '...'}
                  </p>
                )}

                {/* Link */}
                <div className="flex items-center gap-2 text-xs text-gray-400 mb-6" style={{ fontFamily: 'Courier Prime, monospace' }}>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                  </svg>
                  <span className="truncate">
                    {selectedPaper.link ? new URL(selectedPaper.link).hostname : 'Unknown source'}
                  </span>
                </div>

                {/* Action Button */}
                <a
                  href={selectedPaper.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-5 py-2.5 text-sm text-white rounded-lg transition-all hover:shadow-lg hover:-translate-y-0.5 active:translate-y-0"
                  style={{
                    backgroundColor: CLUSTER_THEMES[selectedPaper.cluster].color,
                    fontFamily: 'Courier Prime, monospace'
                  }}
                >
                  <span>Read Resource</span>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Resource Count & Instructions */}
      <div className="absolute bottom-6 right-6 pointer-events-none z-10 text-right">
        <div
          className="text-xs text-gray-300/60 mb-1"
          style={{ fontFamily: 'Courier Prime, monospace' }}
        >
          scroll to zoom · space to reset
        </div>
        <div
          className="text-xs text-gray-300"
          style={{ fontFamily: 'Courier Prime, monospace' }}
        >
          {filteredPapers.length} / {processedPapers.length}
        </div>
      </div>

      {/* Empty State */}
      {filteredPapers.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 flex items-center justify-center">
              <svg className="w-8 h-8 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-gray-400 text-sm" style={{ fontFamily: 'Courier Prime, monospace' }}>
              No papers match your filters
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
