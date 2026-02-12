import Aboutcard from "../components/AboutCard";
import Layout from "../components/Layout";
import Image from "next/image";
import img from "../public/pic.png";

function About() {
  return (
    <Layout>
      <div className="container mx-auto py-16 px-6 sm:px-8 lg:px-12 flex justify-center">
        <div className="w-full max-w-3xl">
          {/* Decorative top line */}
          <div className="w-16 h-px bg-subtle-line mx-auto mb-12"></div>

          <div className="text-deep-charcoal">
            <Aboutcard />
          </div>

          {/* Decorative bottom line */}
          <div className="w-16 h-px bg-subtle-line mx-auto mt-20"></div>
        </div>
      </div>
    </Layout>
  );
}

export default About;
