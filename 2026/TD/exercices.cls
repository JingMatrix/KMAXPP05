% --- Class Identification and Base Class ---
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{../exercices}[Custom class for exercise sheets]
\LoadClass[12pt,a4paper]{article}

% --- Handle Class Options (for toggling solutions) ---
\newif\ifshowsolutions
\DeclareOption{solutions}{\showsolutionstrue}
\ProcessOptions\relax

% --- Required Packages ---
\RequirePackage[left=2cm,right=2cm,top=2cm,bottom=2cm]{geometry}
\pagestyle{empty}

\RequirePackage{amsfonts,amsmath,amssymb,amsthm,amstext}
\RequirePackage{stmaryrd} % For \llbracket

\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{pgf,tikz}
\usetikzlibrary{arrows.meta, arrows}

\RequirePackage{lmodern}
\RequirePackage{multicol}
\RequirePackage{verbatim} % For the comment environment used to hide solutions

% --- Custom Math Commands ---
% Number Sets
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\Var}{\mathbb{V}\mathrm{ar}}
\newcommand{\Cov}{\mathbb{C}\mathrm{ov}}
\newcommand{\sigeps}{\sigma_{\epsilon}^2}

% Symbol Aliases and Redefinitions
\renewcommand{\epsilon}{\varepsilon}
\newcommand{\eps}{\varepsilon}

\definecolor{SolutionGray}{gray}{0.24}

% --- Structure and Environments ---
% Theorem-like environments for exercises and corrections
\theoremstyle{definition}
\newtheorem{exercice}{Exercice}
\newtheorem{cor}{\color{brown}Corrigé}
\ifshowsolutions
	\newenvironment{solution}{
		\begin{cor}
			\hspace{0.cm}
			\sf \color{SolutionGray}
			}{
		\end{cor}
	}
\else
	\newenvironment{solution}{\comment}{\endcomment}
\fi

% --- Header and Title Commands ---
\newcommand{\makeheader}[2]{%
	\noindent{\bfseries Université de Toulouse} \hfill {\bfseries Année #1-#2}
	\vspace{0.5cm}
}

\newcommand{\tdtitle}[2]{%
	\begin{center}
		{\large \bfseries UE Machine Learning - Feuille de TD n°#1} \\
		\vspace{0.25cm}
		{\sf \itshape #2}
	\end{center}
	\hrule
	\vspace{0.3cm}
	\vspace{0.1cm}
}

