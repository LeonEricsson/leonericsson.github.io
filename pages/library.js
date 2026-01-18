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
        <div className="bg-warm-cream" style={{ height: 'calc(100vh - 80px)' }}>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="bg-warm-cream" style={{ height: 'calc(100vh - 80px)' }}>
        <KnowledgeNebula papers={mergedData} />
      </div>
    </Layout>
  );
}
