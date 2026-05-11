#set heading(numbering: none)

#grid(
  columns: (1fr, 1fr),
  align: (left, right),
  image("assets/logo1.jpg", width: 40%),
  image("assets/logo2.jpg", width: 40%)
)

#v(2em)


#block(
  width: 100%,
  fill: black,
  inset: (x: 2em, y: 3em),
)[
  #text(fill: white, size: 22pt, weight: "bold")[
    SPATIO-TEMPORAL CLIMATE PREDICTION \
    ON NASA MODIS LST DATA
  ]
  #v(0.5em)
  #line(length: 15%, stroke: 4pt + rgb("#e03b24"))
  #v(1em)
  #text(fill: rgb("#cccccc"), size: 14pt)[
    Big Data Pipeline & Deep Learning
  ]
]

#v(1em)

#block(
  width: 100%,
  fill: rgb("#f5f5f5"),
  inset: 1.5em,
)[
  #grid(
    columns: (1fr, 1fr),
    row-gutter: 1.5em,
    [
      #text(fill: rgb("#e03b24"), weight: "bold", size: 9pt)[COURSE] \
      Big Data — Practical Task \
      #v(0.5em)
      #text(fill: rgb("#e03b24"), weight: "bold", size: 9pt)[INSTRUCTORS] \
      Dr. Nagwa Yaseen \
      Eng. Hend Maher \
      Eng. Doa Ali
    ],
    [
      #text(fill: rgb("#e03b24"), weight: "bold", size: 9pt)[INSTITUTION] \
      Faculty of Computer Science & Info. Systems \
      #v(0.5em)
      #text(fill: rgb("#e03b24"), weight: "bold", size: 9pt)[DATE] \
      May 2026
    ]
  )
]

#v(2em)

#text(fill: rgb("#e03b24"), weight: "bold", size: 11pt)[TEAM MEMBERS]
#v(0.5em)

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (center, left, left),
    table.header([*\#*], [*Student Name*], [*Student ID*]),
    [1], [Mohamed AHmed Mohamed ALi Shehata], [202203567],
    [2], [Serag Al-Deen Assem Abd El-Azeez], [202202723],
    [3], [Basil Ameen Mohamed Ahmed], [202203981],
    [4], [Hassan Khalaf Hassan], [202203082],
    [5], [Amr Yasser Hemdan Ibrahim], [202203819],
    [6], [Ahmed Hany Fathy], [202200721],
    [7], [Adham Ibrahim Abd El-shafy kasim], [202201929]

  )
)

#v(2em)

= Abstract

This project implements a complete end-to-end Big Data pipeline for spatio-temporal climate prediction over Egypt. We utilize NASA MODIS MOD11C1 Land Surface Temperature data (2014–2024), processed through a MongoDB-backed ETL pipeline. The workflow involves HDF4 ingestion, MapReduce-style spatial analytics, and the construction of 14-day sliding-window ConvLSTM tensors for daily temperature forecasting. Memory constraints in Google Colab are addressed through yearly chunking and memory-mapped lazy loading, enabling robust deep learning training on large-scale geospatial datasets.

#line(length: 100%)
