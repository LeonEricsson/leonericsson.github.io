import React from "react";
import PublicationList from "../components/PublicationList";
import Layout from "../components/Layout";

export default function Publications() {
  return (
    <Layout>
      <div className="container mx-auto pt-4 md:pt-12 md:px-12 lg:pt-24 lg:px-24">
        <h1 className="text-black text-3xl p-4">
          <strong className="purple">a  <sup style={{ fontSize: '50%' }}>hopefully</sup> ever-growing list of publications.</strong>
        </h1>
        <div className="p-2">
          <PublicationList />
        </div>
      </div>
    </Layout>
  );
}
