import React from "react";

function AboutCard() {
  return (
    <div>
      <h2 className="text-black text-2-5xl font-bold pb-3">about me.</h2>
      <p>
        my name is leon ericsson, i live in stockholm with my <i><a href="https://sverigesradio.se/artikel/what-does-it-mean-to-be-a-sambo-in-sweden" style={{ color: 'blue' }}>sambo</a></i>. i've recently
        graduated with a master's degree in cs. i love ml, research, and code. i enjoy food eating and food making, traveling and the gym.
      </p>
      <br />
      <p>
        i work as a research engineer, with an interest in: <b>foundational models</b>, <b>policy learning</b>, <b>medical ai</b>, and <b>programming</b>. if you're curious about my current fixations, skim the blog. 
      </p>
      <h2 className="text-black text-2-5xl font-bold pb-3 pt-10">i like making things.</h2>
      <p>
        mix of research, experiments, and from scratch builds.
        <ul className="list-none ml-4 pt-2">
          <li className="before:content-['-'] before:pr-2">this website, including my <a href="https://leonericsson.github.io/blog" style={{ color: 'blue' }}>blog</a></li>
          <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/alphax" style={{ color: 'blue' }}>alphax</a>, scalable + fast alphazero in jax. facilitates spmd, simd, jit-composition etc </li>
          <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/llmjudge" style={{ color: 'blue' }}>exploring</a> limitations of llm-as-a-judge</li>
          <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/llmcontext" style={{ color: 'blue' }}>llmcontext</a> (fork of <a href="https://github.com/gkamradt/LLMTest_NeedleInAHaystack" style={{ color: 'blue' }}>original</a>), a context window pressure tester of <b>open</b> llms</li>
          <li className="before:content-['-'] before:pr-2">a <a href="https://github.com/LeonEricsson/Ensemble4DFlowNet" style={{ color: 'blue' }}>framework</a> for super-resolution of clinical 4d flow mri. <a href="https://arxiv.org/abs/2311.11819" style={{ color: 'blue' }}>paper</a>.</li>
          <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/AlphaFour" style={{ color: 'blue' }}>alphafour</a> , minimal alphazero for connect four in pure pytorch</li>
          
        </ul>
      </p>
      <h2 className="text-black text-2-5xl font-bold pb-3 pt-10">i like open source.</h2>
        <p>
          these are my favorite, some of which i contribute to.
          <ul className="list-none ml-4 pt-2">
            <li className="before:content-['-'] before:pr-2">ggerganov's <a href="https://github.com/ggerganov/llama.cpp" style={{ color: 'blue' }}>llama.cpp</a> (<a href="https://github.com/ggerganov/llama.cpp/pull/4484" style={{ color: 'blue' }}>#4484</a>)</li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/ml-explore/mlx" style={{ color: 'blue' }}>mlx</a> is an array framework for apple silicon (<a href="https://github.com/ml-explore/mlx-examples/pull/202" style={{ color: 'blue' }}>#202</a> <a href="https://github.com/ml-explore/mlx-examples/pull/237" style={{ color: 'blue' }}>#237</a> <a href="https://github.com/ml-explore/mlx/pull/456" style={{ color: 'blue' }}>#456</a> <a href="https://github.com/ml-explore/mlx-examples/pull/19" style={{ color: 'blue' }}>#19</a> <a href="https://github.com/ml-explore/mlx-examples/pull/276" style={{ color: 'blue' }}>#276</a>) </li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/LeonEricsson/Tensor-Puzzles" style={{ color: 'blue' }}>tensor puzzles</a> and <a href="https://github.com/LeonEricsson/GPU-Puzzles" style={{ color: 'blue' }}>cuda puzzles</a> from sasha rush</li>
            <li className="before:content-['-'] before:pr-2">community driven <a href="https://github.com/ohmyzsh/ohmyzsh" style={{ color: 'blue' }}>ohmyzsh</a></li>
            <li className="before:content-['-'] before:pr-2"><a href="https://github.com/google/jax" style={{ color: 'blue' }}>jax</a>: the lovechild of autograd and xla</li>
          </ul> 
        </p>
      
    </div>
  );
}

export default AboutCard;
