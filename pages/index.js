import React from "react";
import Layout from "../components/Layout";
import Contact from "../components/Contact";
import Hello from "../components/Hello";
import { getSortedPostsData } from '../lib/posts';


export async function getStaticProps() {
  const allPostsData = getSortedPostsData(); 
  return {
    props: {
      allPostsData
    }
  };
}

function App({allPostsData}) {
  return (
    <Layout>
      <div id="home" className="h-screen flex flex-col">
        <div className="container mx-auto pt-12 md:pt-24 md:px-12 lg:pt-40 lg:px-24 flex-grow">
          <div className="flex flex-col items-center justify-center md:flex-row md:items-center md:justify-center flex-wrap mr-10">
            <Hello allPostsData={allPostsData} />
          </div>
        </div>
        <div className="p-12 mb-16">
          <Contact />
        </div>
      </div>
    </Layout>
  );
}


export default App;
