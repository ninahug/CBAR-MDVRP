from pathlib import Path
src=Path('/mnt/data/t-mdvrp_CBAR_PPBRC_frozen_data_protocol.tex')
out=Path('/mnt/data/t-mdvrp_CBAR_PPBRC_public_benchmark_protocol.tex')
s=src.read_text(encoding='utf-8')
old='''E4     & 32 & 60           & 3    & 50             & 100 validation $+$ 400 test \\
\\bottomrule'''
new='''E4     & 32 & 60           & 3    & 50             & 100 validation $+$ 400 test \\
E5-public & 33/11 & 50--360 & 2--9 & 1/50 & 200 validation $+$ 800 test \\
\\bottomrule'''
if old not in s:
    raise SystemExit('suite table anchor not found')
s=s.replace(old,new,1)
old2='''E2 contains three pre-specified carbon-network strata for every size class: a tight-budget/low-capacity stratum, a balanced stratum, and a loose-budget/high-capacity stratum.  E4 uses a replicated $2^4$ design over demand coefficient of variation ($0.15$ or $0.35$), cross-depot correlation ($-0.25$ or $0.55$), boundary-customer share ($0.10$ or $0.30$), and transfer-capacity scale ($0.60$ or $1.40$).  Every generated instance is retained.  Mechanism diagnostics describe the resulting strata but are never used to remove an unfavourable instance.'''
new2=old2+'''  E5 is an external-validity layer based on the public Cordeau MDVRP instances: all 33 instances are used for an unmodified deterministic routing regression, while 11 pre-specified instances spanning 50--360 customers and 2--9 depots receive the frozen stochastic-demand and carbon-account overlay.  The original public coordinates, depots, mean demands, capacities, route-duration limits, and vehicle limits are never perturbed or filtered.'''
s=s.replace(old2,new2,1)
anchor='''\\subsection{Claims-to-evidence discipline}
\\label{subsec:claims_evidence}
'''
if anchor not in s:
    raise SystemExit('claims anchor not found')
e5=r'''
\subsection{E5: public benchmark external validation}
\label{subsec:e5_public_benchmark}

The synthetic suites are necessary for controlled activation of demand risk,
carbon-budget stringency, and transfer-network congestion, but they do not by
themselves establish external routing credibility.  We therefore add a
separate public-benchmark layer based on the 33 Cordeau MDVRP instances
\citep{cordeau1997vrp}.  The public source files are used without changing
customer coordinates, depot coordinates, mean demands, vehicle capacities,
route-duration limits, or maximum vehicle counts.  The conversion script,
source hashes, converted-file hashes, scenario seeds, and solution-file parsing
status are archived with the computational supplement.

\paragraph{E5a: regression to the classical MDVRP.}
The first test removes every paper-specific extension: demand is deterministic,
the carbon and accounting terms are disabled, and the objective is the original
Euclidean route duration.  The routing engine is run on all 33 public instances
and compared with the published best-known solution (BKS).  This test validates
route representation, multi-depot assignment, vehicle counting, capacity,
route duration, and the diversification substrate.  It is not presented as
evidence for the stochastic carbon-budget mechanism.

\begin{table}[!htbp]
\centering
\caption{E5a public MDVRP routing regression.  The final table reports all 33 Cordeau instances; the rows below define the publication output.}
\label{tab:e5a_public_routing}
\begin{tabular}{lrrrrrr}
\toprule
Instance class & Instances & Customers & Depots & Best gap to BKS (\%) & Mean gap (\%) & Feasible runs (\%)\\
\midrule
Small  &  & 50--80   & 2--5 & \multicolumn{3}{c}{\textit{To be filled from frozen runs}}\\
Medium &  & 100      & 2--4 & \multicolumn{3}{c}{\textit{To be filled from frozen runs}}\\
Large  &  & 160--360 & 4--9 & \multicolumn{3}{c}{\textit{To be filled from frozen runs}}\\
\bottomrule
\end{tabular}
\end{table}

\paragraph{E5b: CBAR on public routing backbones.}
The complete CBAR-MDVRP is evaluated on the pre-specified subset
\{p01,p02,p03,p12,p04,p05,p06,p07,p15,p18,p21\}.  For each instance, the
public mean demand is multiplied by mean-corrected correlated lognormal shocks.
The optimisation, validation, and sealed test banks contain 50, 200, and 800
scenarios, respectively.  Customer eligibility remains unrestricted across the
public depots, as in the classical MDVRP.  The internal transfer graph is a
bidirected Euclidean minimum-spanning tree augmented by each depot's nearest
omitted neighbour.  Transfer friction and capacity are deterministic functions
of public depot distance and the reference fleet emission.

When the public BKS route file can be parsed, its route sequence under mean
demand defines the reference load-dependent emission.  Otherwise, a fully
specified nearest-depot angular-sweep fallback is used and explicitly flagged;
it is never called a BKS route.  Corporate budget factors are
$\{0.90,1.00,1.10\}$ and transfer-capacity scales are
$\{0.60,1.00,1.40\}$.  PPBRC, PPBRC-core, SAA-ALNS, and PH-ALNS are compared
under the same time and seed protocol used in E2.  The original MDVRP BKS is
not used to compute an optimality gap for this augmented objective because the
uncertainty, emissions, and accounting recourse change the problem.

\begin{table}[!htbp]
\centering
\caption{E5b algorithm comparison on public Cordeau backbones with the frozen CBAR overlay.}
\label{tab:e5b_public_cbar}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Method & Train objective & Test objective & Test SE & Relative test cost (\%) & External buy & Internal transfer & Cross-depot (\%) & Time (s)\\
\midrule
SAA-ALNS  & \multicolumn{8}{c}{\textit{To be filled from the sealed public-backbone experiment}}\\
PH-ALNS   & \multicolumn{8}{c}{\textit{To be filled from the sealed public-backbone experiment}}\\
PPBRC-core& \multicolumn{8}{c}{\textit{To be filled from the sealed public-backbone experiment}}\\
PPBRC     & \multicolumn{8}{c}{\textit{To be filled from the sealed public-backbone experiment}}\\
\bottomrule
\end{tabular}}
\end{table}

The two public tests answer different questions.  E5a measures compatibility
with a recognised routing benchmark and can be compared with published BKS
values.  E5b measures algorithmic and policy performance for the new CBAR
problem on externally supplied spatial and capacity structures.  A favourable
E5a result cannot validate the carbon model, and an E5b objective cannot be
compared numerically with the original MDVRP BKS.

'''
s=s.replace(anchor,e5+anchor,1)
old3='''E1 can establish numerical consistency and small-instance solution quality but cannot establish scalability. E2 can identify the contribution of price-polyhedral coordination under the tested instances but cannot prove dominance on all instances. E3 quantifies the value of stochastic optimisation under the chosen scenario model and does not establish robustness to misspecified distributions. E4 identifies empirical complementarity or substitution and does not prove a universal sign for $I$.'''
new3=old3+''' E5a establishes routing compatibility with a public MDVRP benchmark but does not validate the carbon-account extension. E5b provides external spatial and capacity backbones for the augmented model but does not inherit the original MDVRP BKS because its objective and information structure are different.'''
s=s.replace(old3,new3,1)
out.write_text(s,encoding='utf-8')
print(out)
