import Aboutcard from "../components/AboutCard";
import Layout from "../components/Layout";
import Image from "next/image";
import img from "../public/pic.png";

function About() {
  return (
    <Layout>
      <div className="container mx-auto py-12 px-4 sm:px-6 lg:px-8 flex justify-center">
        <div className="w-full lg:w-3/5">
          <div className="px-12">
            <Aboutcard />
          </div>
          <div className="px-12 pt-12">
            <Image
              src={img}
              alt="me and my beautiful sambo"
              className="w-5/6 h-auto ml-[20px]"
              width="2000"
              height="2000"
            />
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default About;
