import React, { useMemo, useState, useEffect } from 'react';
import Layout from '../components/Layout';
import knowledgeGraph from '../data/knowledge_graph.json';
import readingList from '../data/reading_list.json';
import dynamic from 'next/dynamic';

// Dynamically import the visualization to avoid SSR issues with canvas
const KnowledgeNebula = dynamic(
  () => import('../components/KnowledgeNebula'),
  { ssr: false }
);

export default function Library() {
  const [isClient, setIsClient] = useState(false);

  // Merge knowledge graph positions with reading list metadata
  const mergedData = useMemo(() => {
    const readingMap = new Map(readingList.map(item => [item.id, item]));
    return knowledgeGraph.map(node => ({
      ...node,
      ...readingMap.get(node.id),
    }));
  }, []);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) {
    return (
      <Layout>
        <div className="bg-warm-cream min-h-screen">
          <div className="container mx-auto py-16 px-6 sm:px-8 lg:px-12">
            {/* Decorative top line */}
            <div className="w-16 h-px bg-subtle-line mx-auto mb-12"></div>

            <div className="max-w-3xl mx-auto text-center">
              <h1 className="text-4xl font-sorts-mill italic text-deep-charcoal mb-6">
                Library
              </h1>

              <p className="text-medium-gray font-lora text-base leading-relaxed mb-12">
                Loading constellation...
              </p>

              <div className="flex items-center justify-center">
                <div className="w-8 h-8">
                  <svg className="animate-spin w-full h-full text-sepia-accent" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                </div>
              </div>
            </div>

            {/* Decorative bottom line */}
            <div className="w-16 h-px bg-subtle-line mx-auto mt-12"></div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="bg-warm-cream">
        <div className="container mx-auto pt-12 pb-6 px-6 sm:px-8 lg:px-12">
          {/* Decorative top line */}
          <div className="w-16 h-px bg-subtle-line mx-auto mb-10"></div>

          <div className="max-w-3xl mx-auto text-center mb-8">
            <h1 className="text-5xl font-sorts-mill italic text-deep-charcoal mb-6">
              Library
            </h1>

            <p className="text-medium-gray font-lora text-base leading-relaxed max-w-2xl mx-auto">
              an interactive constellation of readings that have shaped my thinking.
              each node represents a paper, article, or book—connected by themes,
              ideas, and intellectual threads. explore the nebula to discover
              relationships between works and trace the evolution of concepts.
            </p>
          </div>

          {/* Decorative separator */}
          <div className="w-16 h-px bg-subtle-line mx-auto mb-8"></div>
        </div>

        <div className="px-4" style={{ height: 'calc(100vh - 320px)' }}>
          <KnowledgeNebula papers={mergedData} />
        </div>
      </div>
    </Layout>
  );
}
