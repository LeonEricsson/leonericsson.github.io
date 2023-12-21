import React from "react";
import PublicationList from "../components/PublicationList";
import Layout from "../components/Layout";
import { ArrowRightCircleIcon } from "@heroicons/react/24/outline";

export default function Publications() {
  return (
    <Layout>
      <div className="container mx-auto pt-4 md:pt-12 md:px-12 lg:pt-24 lg:px-24">
        <h1 className="text-white text-4xl p-4">
          <strong className="purple">Selected Publications </strong>
        </h1>
        <div className="p-4">
          <PublicationList />
        </div>
      </div>
    </Layout>
  );
}
