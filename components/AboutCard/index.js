import React from "react";

function AboutCard() {
  return (
    <div>
      <h2 className="text-black text-2-5xl font-bold pb-3">about me.</h2>
      <p>
        my name is leon ericsson. i'm a recent grad with a master's degree in machine
        learning living in Stockholm with my <i><a href="https://collectum.se/en/startpage/private/your-situation/i-have-a-sambo#:~:text=%E2%80%9CSambo%E2%80%9D%20is%20a%20Swedish%20legal,cover%20and%2For%20Repayment%20cover." style={{ color: 'blue' }}>sambo</a></i>.
        i like to workout, cook, travel, and i hate walking.
      </p>
      <br />
      <p>
        i'm currently employed as a research engineer; my research interests are transient,
        feels like I find something new every other week, but broadly speaking: 
        <b> foundational models</b>, <b>policy learning</b>, <b>medical ai</b>, and <b>coding</b>.
        If you're curious about my current concentrations, skim my blog. I hacked this website as a platform to share
        and document my thoughts on research that I come across. While primarily for my own edification,
        hopefully there's something here that proves insightful to you.
      </p>
      <h2 className="text-black text-2-5xl font-bold pb-3 pt-10">i like making things.</h2>
      <p>
        most of my stuff is educational, but i'm moving towards more practical projects.
        <ul className="list-none ml-4 pt-2">
          <li className="before:content-['-'] before:pr-2">this website, including my <a href="https://leonericsson.github.io/blog" style={{ color: 'blue' }}>blog</a></li>
          <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/llmcontext" style={{ color: 'blue' }}>llmcontext</a> (fork of <a href="https://github.com/gkamradt/LLMTest_NeedleInAHaystack" style={{ color: 'blue' }}>original</a>), a context window pressure tester of <b>open</b> llms</li>
          <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/AlphaFour" style={{ color: 'blue' }}>alphafour</a> , scratch implementation of alphazero for connect four in pure pytorch</li>
          <li className="before:content-['-'] before:pr-2">a <a href="https://github.com/LeonEricsson/Ensemble4DFlowNet" style={{ color: 'blue' }}>framework</a> for super-resolution of clinical 4d flow mri. <a href="https://arxiv.org/abs/2311.11819" style={{ color: 'blue' }}>paper</a>.</li>
        </ul>
      </p>
      <h2 className="text-black text-2-5xl font-bold pb-3 pt-10">i like open source.</h2>
        <p>
          these are my favorite, in no particular order
          <ul className="list-none ml-4 pt-2">
            <li className="before:content-['-'] before:pr-2">ggerganov's <a href="https://github.com/ggerganov/llama.cpp" style={{ color: 'blue' }}>llama.cpp</a> (<a href="https://github.com/ggerganov/llama.cpp/pull/4484" style={{ color: 'blue' }}>#4484</a>)</li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/ml-explore/mlx" style={{ color: 'blue' }}>mlx</a> is an array framework for apple silicon (<a href="https://github.com/ml-explore/mlx-examples/pull/202" style={{ color: 'blue' }}>#202</a> <a href="https://github.com/ml-explore/mlx-examples/pull/19" style={{ color: 'blue' }}>#19</a>) </li>
            <li className="before:content-['-'] before:pr-2"> <a href="https://github.com/srush/Tensor-Puzzles" style={{ color: 'blue' }}>tensor puzzles</a> from sasha rush</li>
            <li className="before:content-['-'] before:pr-2"> community driven <a href="https://github.com/ohmyzsh/ohmyzsh" style={{ color: 'blue' }}>ohmyzsh</a></li>
            <li className="before:content-['-'] before:pr-2"> <a href="https://github.com/google/jax" style={{ color: 'blue' }}>jax</a>: the lovechild of autograd and xla</li>
          </ul> 
        </p>
      
    </div>
  );
}

export default AboutCard;
