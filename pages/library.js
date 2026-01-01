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
        <div
          className="flex items-center justify-center"
          style={{ height: 'calc(100vh - 80px)', backgroundColor: 'rgb(255, 254, 249)' }}
        >
          <div className="text-center">
            <div className="w-8 h-8 mx-auto mb-4">
              <svg className="animate-spin w-full h-full text-gray-300" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            </div>
            <p className="text-gray-400 text-sm" style={{ fontFamily: 'Courier Prime, monospace' }}>
              Loading library...
            </p>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div style={{ height: 'calc(100vh - 80px)', backgroundColor: 'rgb(255, 254, 249)' }}>
        <KnowledgeNebula papers={mergedData} />
      </div>
    </Layout>
  );
}
