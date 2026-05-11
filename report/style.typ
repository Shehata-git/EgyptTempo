#let conf(title: none, doc) = {
  set document(title: title)
  set page(paper: "a4", margin: 2.5cm)
  set text(font: "Noto Sans", size: 10pt, fill: rgb("#222222"))
  set heading(numbering: "1.1.")
  
  let amd-red = rgb("#e03b24")
  
  show heading.where(level: 1): it => {
    v(1.5em)
    block(
      stroke: (left: 3pt + amd-red),
      inset: (left: 0.5em),
      text(fill: black, weight: "bold", size: 16pt, it)
    )
    v(0.5em)
  }

  show heading.where(level: 2): it => {
    v(1em)
    grid(
      columns: (auto, 1fr),
      column-gutter: 0.5em,
      align(horizon)[#rect(width: 0.5em, height: 0.5em, fill: amd-red)],
      align(horizon)[#text(fill: black, weight: "bold", size: 12pt, it)]
    )
    v(0.5em)
  }

  show raw: set text(font: "Noto Sans", size: 9pt, fill: rgb("#333333"))
  
  set table(
    stroke: 0.5pt + rgb("#d0d0d0"),
    fill: (col, row) => {
      if row == 0 {
        black
      } else if calc.rem(row, 2) == 0 {
        rgb("#f5f5f5")
      } else {
        white
      }
    }
  )
  
  show table.cell: it => {
    if it.y == 0 {
      set text(fill: white, weight: "bold")
      it
    } else {
      it
    }
  }

  show figure: set block(breakable: true)
  
  doc
}