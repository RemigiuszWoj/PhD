"""Build overleaf/main.tex (supervisor's layout) from response_to_reviewers.tex.

Revised text of the manuscript is quoted below each response (the "Before" text is not printed).
Green: answers and notes to be completed after the repeated experiments.
Run from response/: python3 overleaf/make_overleaf.py
"""
import re

S = open('response_to_reviewers.tex').read()
body = S[S.index('\\section*{Response to the reviewers}'):S.index('\\end{document}')]
intro, rest = body.split('% =====================================================================\n\\newpage\n\\section*{Answers to the Reviewer \\#1 remarks}', 1)
intro = intro.replace('\\section*{Response to the reviewers}\n', '')
intro = re.sub(r'\\todo\{2--3 sentences[^}]*\}', r'\\upd{[Two or three sentences on the main changes will be added after the repeated experiments.]}', intro)
intro = intro.replace(
    "Each point quotes the comment in italics. Our response follows directly below it and states where the manuscript was changed.",
    "Each point quotes the comment in italics. Our response follows directly below it and states where the manuscript was changed; the revised text of the manuscript is quoted below the response.")
intro = re.sub(r'Below, the comments of each reviewer are divided into numbered points: R1\.1--R1\.6 for Reviewer~\\#1 and R2\.\\,\\dots\\ for Reviewer~\\#2, following the structure of the review\. ',
               'Below, the comments of each reviewer are quoted point by point, following the structure of the review. ', intro)
r1, r2 = rest.split('\\section*{Answers to the Reviewer \\#2 remarks}', 1)


def hdr(k):
    return ('\\begin{center}\n{\\bf Answers to the Reviewer \\#%d remarks\\\\[2mm]\n'
            '``Quantum and Classical Annealing for Very Large-Scale Neighborhood Search in the Permutation Flow Shop Problem\'\'}\\\\\n'
            'BPASTS-05673-2026-02\\\\[5mm]\n\\end{center}\n' % k)


pre = r'''\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage[margin=2.5cm]{geometry}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{algpseudocode}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,decorations.pathreplacing}

%\usepackage{polski}
%\usepackage{todonotes}

\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlist{itemsep=1pt,topsep=2pt}

% Reviewer text italic, our text upright. Green: still open.
\newcommand{\upd}[1]{\textcolor{green!50!black}{#1}}
% reviewer comment (italic, emphasis bold italic); the argument is an internal label and is not printed
\newenvironment{rc}[1]{\par\medskip\noindent\itshape\renewcommand{\emph}[1]{\textbf{##1}}\ignorespaces}{\par}
\newenvironment{lead}{\par\medskip\noindent\itshape\renewcommand{\emph}[1]{\textbf{##1}}\ignorespaces}{\par}
\newcommand{\resp}{\par\smallskip\noindent\upd{[The answer to this point will be completed after the repeated experiments.]}\par\medskip}
\newcommand{\answer}[1]{\par\smallskip\noindent #1\par}
% change in the manuscript: where, previous text (not printed), revised text
\newcommand{\change}[3]{\par\smallskip\noindent\textbf{Revised text (#1).}\par
  {\leftskip=1.5em\noindent#3\par}}
\newcommand{\pending}[1]{\par\smallskip\noindent\upd{[This answer will be updated with the results of the repeated experiments.]}\par}
\newcommand{\todo}[1]{\upd{[#1]}}
\newcommand{\Cmax}{C_{\max}}

\begin{document}

'''
# the supervisor's bibliography, verbatim
bib = ('\\begin{thebibliography}{10}\n\n\\bibitem{bozejko2022}\nBożejko, W., Pempera, J., Uchroński, M., Wodecki, M. (2022). '
       'Distributed Quantum Annealing on D-Wave for the Single Machine Total Weighted Tardiness Scheduling Problem. In: Groen, D., '
       'de Mulatier, C., Paszynski, M., Krzhizhanovskaya, V.V., Dongarra, J.J., Sloot, P.M.A. (eds) Computational Science – ICCS 2022. '
       'LNCS 13353, Springer, p. 171--178.\n\n\n\\end{thebibliography}\n\n\n\\end{document}\n\n')
out = pre + hdr(1) + '\n' + intro.strip() + '\n\n' + r1.strip() + '\n\n\\newpage\n\n' + hdr(2) + '\n' + r2.strip() + '\n\n' + bib
open('overleaf/main.tex', 'w').write(out)
print(len(out))
