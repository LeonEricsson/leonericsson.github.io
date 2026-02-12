import React, { useMemo, useState, useEffect } from 'react';
import Layout from '../components/Layout';
import knowledgeGraph from '../data/knowledge_graph.json';
import readingList from '../data/reading_list.json';
import KnowledgeConstellationWrapper from '../components/KnowledgeConstellation';

export default function Readings() {
  const [isClient, setIsClient] = useState(false);

  // Merge knowledge graph with reading list data
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
        <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: 'rgb(255, 254, 249)' }}>
          <div className="text-gray-400 font-mono">Loading...</div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="min-h-screen" style={{ backgroundColor: 'rgb(255, 254, 249)' }}>
        <KnowledgeConstellationWrapper
          mergedData={mergedData}
          readingList={readingList}
        />
      </div>
    </Layout>
  );
}
