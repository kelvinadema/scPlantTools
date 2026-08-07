use seq_geom_parser::{parse_geometry, validate_geometry, CompiledGeom};

fn show(label: &str, geom: &str, r1: &[u8], r2: &[u8], expect_bc: &[u8]) {
    println!("=== {label} ===");
    println!("  geometry : {geom}");
    let parsed = match parse_geometry(geom) {
        Ok(p) => p,
        Err(e) => {
            println!("  PARSE FAILED: {:?}", e.len());
            return;
        }
    };
    match validate_geometry(&parsed) {
        Ok(()) => println!("  validate : OK"),
        Err(e) => {
            println!("  VALIDATE FAILED: {e}");
            return;
        }
    }
    let compiled = match CompiledGeom::from_fragment_geom(&parsed) {
        Ok(c) => c,
        Err(e) => {
            println!("  COMPILE FAILED: {e}");
            return;
        }
    };
    let meta = compiled.meta();
    println!(
        "  meta     : num_bc_levels={} bc_lens={:?} umi_len={}",
        meta.num_bc_levels, meta.bc_lens, meta.umi_len
    );

    let seqs = compiled.extract(r1, r2);
    for (i, bc) in seqs.barcodes.iter().enumerate() {
        println!(
            "  bc[{i}]    : {:?} (len {})",
            bc.map(|b| String::from_utf8_lossy(b).to_string()),
            bc.map(|b| b.len()).unwrap_or(0)
        );
    }
    println!(
        "  umi      : {:?}",
        seqs.umi.map(|b| String::from_utf8_lossy(b).to_string())
    );
    let got = seqs.barcodes.first().copied().flatten().unwrap_or(b"");
    println!("  EXPECTED : {}", String::from_utf8_lossy(expect_bc));
    println!("  GOT      : {}", String::from_utf8_lossy(got));
    println!(
        "  VERDICT  : {}\n",
        if got == expect_bc {
            "correct"
        } else {
            ">>> WRONG <<<"
        }
    );
}

fn main() {
    // BD Rhapsody Enhanced Bead V2 synthetic R1:
    //   offset(2) CLS1(9) GTGA CLS2(9) GACA CLS3(9) UMI(8)
    let cls1 = b"AAAAAAAAA";
    let cls2 = b"CCCCCCCCC";
    let cls3 = b"TTTTTTTTT";
    let umi = b"GGGGTTTT";
    let mut r1 = Vec::new();
    r1.extend_from_slice(b"GG"); // 2bp diversity offset
    r1.extend_from_slice(cls1);
    r1.extend_from_slice(b"GTGA");
    r1.extend_from_slice(cls2);
    r1.extend_from_slice(b"GACA");
    r1.extend_from_slice(cls3);
    r1.extend_from_slice(umi);
    let r2 = b"ACGTACGTACGTACGTACGTACGTACGTACGT";

    let mut expect = Vec::new();
    expect.extend_from_slice(cls1);
    expect.extend_from_slice(cls2);
    expect.extend_from_slice(cls3);

    println!("R1 = {}\n", String::from_utf8_lossy(&r1));

    show(
        "BD Rhapsody Enhanced V2 (user's geometry)",
        "1{x[0-3]b[9]f[GTGA]b[9]f[GACA]b[9]u[8]}2{r:}",
        &r1,
        r2,
        &expect,
    );

    // Control: no anchors, no offset -- plain 3-piece concatenation
    let mut r1b = Vec::new();
    r1b.extend_from_slice(cls1);
    r1b.extend_from_slice(cls2);
    r1b.extend_from_slice(cls3);
    r1b.extend_from_slice(umi);
    show(
        "Plain 3-piece barcode (no anchors)",
        "1{b[9]b[9]b[9]u[8]}2{r:}",
        &r1b,
        r2,
        &expect,
    );

    // Control: known-good single-piece chemistry (10x v3)
    let mut r1c = Vec::new();
    r1c.extend_from_slice(b"ACGTACGTACGTACGT");
    r1c.extend_from_slice(b"AAAAAAAAAAAA");
    r1c.extend_from_slice(b"TTTTTT");
    show(
        "10x Chromium v3 (control, should be correct)",
        "1{b[16]u[12]x:}2{r:}",
        &r1c,
        r2,
        b"ACGTACGTACGTACGT",
    );
}
