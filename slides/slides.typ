#import "@preview/definitely-not-isec-slides:1.0.1": *
#import "@preview/subpar:0.2.2"

#show: definitely-not-isec-theme.with(
  aspect-ratio: "16-9",
  slide-alignment: top,
  progress-bar: true,
  institute: [Network Science 2025],
  logo: [#tugraz-logo],
  config-info(
    title: [Analysis of brain blood supply and it's resistance to Thrombosis],
    subtitle: [Empirical Analysis of the Dataset],
    authors: ([Denis~Dagbert], [Tomáš~Kysela], [Hussain~Miraah~Rasheed], [Muhammad~Zubair]),
    extra: [],
    footer: [Denis~Dagbert, Tomáš~Kysela, Hussain~Miraah~Rasheed, Muhammad~Zubair],
    download-qr: "https://github.com/tomkys144/NetSci-Project/raw/refs/heads/main/slides/slides.pdf",
  ),
  config-common(
    handout: true,
  ),
  config-colors(
    primary: rgb("e4154b"),
  ),
)

#set cite(style: "ieee")

#title-slide()


#slide(title: [Dataset])[
  - Whole brain vasculature of CD1-Elite mouse
  - Already preprocessed @suprosanna_2021_5367262
  - Dataset used is CD1-E-2 @Todorov2020311
  - 1 664 811 nodes and 2 150 326 edges
  - Data on the diameter, size and volume of the blood vessels
  - Avg radius used as edge weight has lognorm distribution with $s = 0.49$

  #grid(
    columns: (1fr, 1fr),
    figure(
      image(
        "img/graph-CD1-E_no2-xy.png",
        width: 40%,
      ),
      caption: [XY cut of the graph],
    ),
    figure(
      image(
        "img/pdf-edges-CD1-E_no2.png",
        width: 54%,
      ),
      caption: [Distribution of edge avg. radius],
    ),
  )
]

#slide(title: "Centralities")[
  - Stepped PageRank corresponds with hierarchy of blood supply
  - Eigenvector centrality shows major arteries with most of nodes being on the branches
  - Degree centrality shows that most junctions are bifurcations, which is consistent with vascular systems
  #subpar.grid(
    columns: (1fr, 1fr, 1fr),
    figure(
      image(
        "img/cdf-degree-CD1-E_no2.png",
        width: 70%,
      ),
      caption: [Degree],
    ),
    figure(
      image(
        "img/cdf-eigenvector-CD1-E_no2.png",
        width: 70%,
      ),
      caption: [Eigenvector centrality],
    ),
    figure(
      image(
        "img/cdf-pagerank-CD1-E_no2.png",
        width: 70%,
      ),
      caption: [PageRank],
    ),

    caption: [centrality CDFs],
  )
]

#slide(title: "Clustering")[
  - Global clustering ~26,500x random graph confirms Small-World" architecture
  - Low density + high clustering indicates tightly concentrated neighborhoods
  - Log-scale histogram reveals fully connected subgroups (cliques)

  #subpar.grid(
    columns: (1fr, 1fr),
    figure(
      image(
        "img/clustering-local-CD1-E_no2.png",
        height: 55%,
      ),
      caption: [Linear scale],
    ),
    figure(
      image(
        "img/clustering-local-CD1-E_no2-log.png",
        height: 55%,
      ),
      caption: [Log-scale],
    ),

    caption: [Clustering coefficient histograms],
  )
]

#slide(title: "Communities")[
  - Nested stochastic block model
  - 2022 communities in layer 0
  - Layer 3 roughly corresponds to Brodmann Areas
  - Layer 7 with only 2 communities does not correspond to hemispheres

  #grid(
    columns: 1fr,
    figure(
      image(
        "img/sbm-CD1-E_no2_radial.png",
        height: 50%,
      ),
      caption: [Radial view of communities],
    ),
  )
]

#slide(title: [Bibliography])[
  #bibliography("bibliography.bib", style: "ieee")

  #v(1fr)

  #text(size: 24pt, weight: "semibold")[Acknowledgments]

  #image(
    "gh.svg",
    width: 30%,
  )
  Special thanks to #link("https://gamerhost.pro/")[Gamerhost.pro] for providing computational power.
]
