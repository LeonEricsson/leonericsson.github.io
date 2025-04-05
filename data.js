import { AiOutlineNodeCollapse } from "react-icons/ai";
import { FaLinkedinIn, FaAws } from "react-icons/fa";
import { BsMedium } from "react-icons/bs";
import {
  SiGooglescholar,
  SiTensorflow,
  SiApachekafka,
  SiVault,
  SiPytorch,
  SiGrafana,
  SiDocker,
  SiKubernetes,
  SiNginx,
  SiNvidia,
  SiGoogleoptimize,
  SiMongodb,
  SiMysql,
  SiPrometheus,
  SiCypress,
  SiNextdotjs,
  SiGoland,
} from "react-icons/si";
import {
  DiJavascript1,
  DiReact,
  DiNodejs,
  DiDjango,
  DiPython,
  DiGit,
  DiJava,
} from "react-icons/di";
import { CgCPlusPlus } from "react-icons/cg";
import { MdOutlineWeb } from "react-icons/md";

import {FiMail, FiGithub, FiLinkedin} from "react-icons/fi"

export const contacts = [
  {
    url: "https://github.com/LeonEricsson",
    icon: FiGithub,
  },
  {
    url: "mailto:leon.ericsson@icloud.com",
    icon: FiMail,
  },
];

export const skills = [
  {
    category: "Deep Learning",
    items: [
      {
        name: "Python",
        icon: DiPython,
      },
      {
        name: "C++",
        icon: CgCPlusPlus,
      },
      {
        name: "PyTorch",
        icon: SiPytorch,
      },
      {
        name: "TensorFlow",
        icon: SiTensorflow,
      },
      {
        name: "CUDA",
        icon: SiNvidia,
      },
      {
        name: "Optimization",
        icon: SiGoogleoptimize,
      },
    ],
  },
  {
    category: "System",
    items: [
      {
        name: "Git",
        icon: DiGit,
      },
      {
        name: "Kafka",
        icon: SiApachekafka,
      },
      {
        name: "Vault",
        icon: SiVault,
      },
      {
        name: "Grafana",
        icon: SiGrafana,
      },
      {
        name: "Prometheus",
        icon: SiPrometheus,
      },
      { name: "Docker", icon: SiDocker },
      {
        name: "Kubernetes",
        icon: SiKubernetes,
      },
      {
        name: "CI/CD",
        icon: AiOutlineNodeCollapse,
      },
      {
        name: "MySQL",
        icon: SiMysql,
      },
      {
        name: "MongoDB",
        icon: SiMongodb,
      },
      {
        name: "Nginx",
        icon: SiNginx,
      },
      {
        name: "AWS",
        icon: FaAws,
      },
    ],
  },
  {
    category: "Web Development",
    items: [
      {
        name: "JavaScript",
        icon: DiJavascript1,
      },
      {
        name: "Golang",
        icon: SiGoland,
      },
      {
        name: "Java",
        icon: DiJava,
      },
      {
        name: "React",
        icon: DiReact,
      },
      {
        name: "Node.js",
        icon: DiNodejs,
      },
      {
        name: "Django",
        icon: DiDjango,
      },
      {
        name: "Next.js",
        icon: SiNextdotjs,
      },
      {
        name: "Micro-frontend",
        icon: MdOutlineWeb,
      },
      {
        name: "Cypress",
        icon: SiCypress,
      },
    ],
  },
];

export const publications = [
  {
    category: "Computer Vision",
    items: [
      {
        name: "Generalized super-resolution 4D Flow MRI - using ensemble learning to extend across the cardiovascular system",
        authors: [
          "Leon Ericsson", "Adam Hjalmarsson", "Muhammad Usman Akbar", 
          "Edward Ferdian", "Mia Bonini", "Brandon Hardy", 
          "Jonas Schollenberger", "Maria Aristova", "Patrick Winter", 
          "Nicholas Burris", "Alexander Fyrdahl", "Andreas Sigfridsson", 
          "Susanne Schnell", "C. Alberto Figueroa", "David Nordsletten", 
          "Alistair A. Young", "David Marlevi"],
        publisher:
          "Pending review",
        url: "https://arxiv.org/abs/2311.11819",
      },
    ],
  },
];
