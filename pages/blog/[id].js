import { getAllPostIds, getPostData } from "../../lib/posts";
import Layout from "../../components/Layout";
import { Raleway } from "@next/font/google";
import Head from "next/head";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css"; // `rehype-katex` does not import the CSS for you

const raleway = Raleway({
  weight: ['400', '700'],
  style: ['normal', 'italic'],
  subsets: ["latin"],
});

export async function getStaticPaths() {
  const paths = getAllPostIds();
  return {
    paths,
    fallback: false,
  };
}

export async function getStaticProps({ params }) {
  const postData = await getPostData(params.id);
  return {
    props: {
      postData,
    },
  };
}

const formatDate = (dateString) => {
  const options = { year: "numeric", month: "long", day: "numeric" };
  return new Date(dateString).toLocaleDateString("en-US", options);
};

export default function Post({ postData }) {
  return (
    <Layout>
      <Head></Head>

      <article className="container mx-auto py-12 px-4 sm:px-6 lg:px-8 flex justify-center">
        <div className="w-full lg:w-3/5">
          {" "}
          {/* Same proportions as in Blog component */}
          <h1 className="text-3xl font-bold mb-4 text-center">
            {postData.title}
          </h1>
          <p className="text-gray-400 text-base text-center pb-4 pt-1">
            {postData.author ? (
              <>
                Original{" "}
                <a
                  href={postData.exturl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300"
                >
                  paper
                </a>{" "}
                · {postData.author} et al {postData.year}
              </>
            ) : (
              <>{formatDate(postData.date)}.</>
            )}
          </p>
          <div className={`entry text-base mb-4 ${raleway.className}`}>
            <Markdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex, rehypeRaw]}
              className="markdown"
            >
              {postData.contentMarkdown}
            </Markdown>
          </div>
        </div>
      </article>
    </Layout>
  );
}
