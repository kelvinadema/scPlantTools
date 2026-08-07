use seq_geom_parser::{parse_geometry, validate_geometry, CompiledGeom, MAX_BC_TOTAL_LEN};
use smallvec::SmallVec;

const BD_V2: &str = "1{x[0-3]b[9]f[GTGA]b[9]f[GACA]b[9]u[8]}2{r:}";

fn build_r1(offset: &[u8], cls1: &[u8], cls2: &[u8], cls3: &[u8], umi: &[u8]) -> Vec<u8> {
    let mut r1 = Vec::new();
    r1.extend_from_slice(offset);
    r1.extend_from_slice(cls1);
    r1.extend_from_slice(b"GTGA");
    r1.extend_from_slice(cls2);
    r1.extend_from_slice(b"GACA");
    r1.extend_from_slice(cls3);
    r1.extend_from_slice(umi);
    r1
}

fn compile(geom: &str) -> Result<CompiledGeom, String> {
    let parsed = parse_geometry(geom).map_err(|e| format!("parse failed ({} errors)", e.len()))?;
    validate_geometry(&parsed)?;
    CompiledGeom::from_fragment_geom(&parsed)
}

fn extract_bc(compiled: &CompiledGeom, r1: &[u8], r2: &[u8]) -> (String, usize, usize) {
    let seqs = compiled.extract(r1, r2);
    let bc = &seqs.barcodes[0];
    let mut buf: SmallVec<[u8; MAX_BC_TOTAL_LEN]> = SmallVec::new();
    let assembled = bc.assemble_into(&mut buf);
    (
        String::from_utf8_lossy(assembled).to_string(),
        bc.len(),
        bc.pieces().len(),
    )
}

fn check(label: &str, got: &str, want: &str) -> bool {
    let ok = got == want;
    println!(
        "  {:<44} {}  (got {})",
        label,
        if ok { "PASS" } else { "FAIL" },
        got
    );
    ok
}

fn main() {
    let mut all_ok = true;
    let r2 = b"ACGTACGTACGTACGTACGTACGTACGTACGT";

    // ---- 1. The BD Rhapsody Enhanced Bead V2 geometry ------------------
    println!("1. BD Rhapsody Enhanced Bead V2: {BD_V2}");
    let compiled = compile(BD_V2).expect("BD geometry must compile");
    let meta = compiled.meta();
    println!(
        "   meta: num_bc_levels={} bc_lens={:?} umi_len={}",
        meta.num_bc_levels, meta.bc_lens, meta.umi_len
    );
    all_ok &= meta.bc_lens[0] == 27;
    println!(
        "  {:<44} {}",
        "declared barcode width is 27",
        if meta.bc_lens[0] == 27 { "PASS" } else { "FAIL" }
    );

    // Every diversity-offset length 0..=3 must yield the same cell label.
    for (i, offset) in [b"".as_slice(), b"G", b"GG", b"GGT"].iter().enumerate() {
        let r1 = build_r1(offset, b"AAAAAAAAA", b"CCCCCCCCC", b"TTTTTTTTT", b"GGGGTTTT");
        let (bc, len, pieces) = extract_bc(&compiled, &r1, r2);
        all_ok &= check(
            &format!("offset len {i} -> 27bp label ({pieces} pieces, len {len})"),
            &bc,
            "AAAAAAAAACCCCCCCCCTTTTTTTTT",
        );
    }

    // UMI must still be correct.
    let r1 = build_r1(b"GG", b"AAAAAAAAA", b"CCCCCCCCC", b"TTTTTTTTT", b"GGGGTTTT");
    let seqs = compiled.extract(&r1, r2);
    all_ok &= check(
        "UMI",
        &String::from_utf8_lossy(seqs.umi.unwrap()),
        "GGGGTTTT",
    );
    all_ok &= check(
        "biological read (R2)",
        &String::from_utf8_lossy(seqs.reads[0]),
        &String::from_utf8_lossy(r2),
    );

    // ---- 2. Distinctness: cells differing only in block 1 --------------
    // This is what the old behaviour destroyed: it kept only block 3, so
    // these two cells collapsed into one barcode.
    println!("\n2. Cells differing only in block 1 must stay distinct");
    let a = build_r1(b"GG", b"AAAAAAAAA", b"CCCCCCCCC", b"TTTTTTTTT", b"GGGGTTTT");
    let b = build_r1(b"GG", b"GGGGGGGGG", b"CCCCCCCCC", b"TTTTTTTTT", b"GGGGTTTT");
    let (bc_a, _, _) = extract_bc(&compiled, &a, r2);
    let (bc_b, _, _) = extract_bc(&compiled, &b, r2);
    let distinct = bc_a != bc_b;
    all_ok &= distinct;
    println!(
        "  {:<44} {}  ({} vs {})",
        "distinct barcodes",
        if distinct { "PASS" } else { "FAIL" },
        bc_a,
        bc_b
    );

    // ---- 3. Single-piece chemistries stay zero-copy --------------------
    println!("\n3. Single-piece chemistry unchanged and zero-copy");
    let v3 = compile("1{b[16]u[12]x:}2{r:}").unwrap();
    let mut r1c = Vec::new();
    r1c.extend_from_slice(b"ACGTACGTACGTACGT");
    r1c.extend_from_slice(b"AAAAAAAAAAAA");
    r1c.extend_from_slice(b"TTTTTT");
    let seqs = v3.extract(&r1c, r2);
    let contiguous = seqs.barcodes[0].as_contiguous();
    all_ok &= check(
        "10x v3 barcode",
        &String::from_utf8_lossy(contiguous.unwrap()),
        "ACGTACGTACGTACGT",
    );
    // Borrowed directly from the read: no copy was made.
    let borrows_read = contiguous.unwrap().as_ptr() == r1c.as_ptr();
    all_ok &= borrows_read;
    println!(
        "  {:<44} {}",
        "as_contiguous() borrows the read (no copy)",
        if borrows_read { "PASS" } else { "FAIL" }
    );

    // ---- 4. Validation guards ------------------------------------------
    println!("\n4. Validation guards on split levels");
    let cases: [(&str, &str, &str); 3] = [
        (
            "too wide (4x9 = 36 > 32)",
            "1{b[9]f[GTGA]b[9]f[GTGA]b[9]f[GTGA]b[9]u[8]}2{r:}",
            "totals",
        ),
        (
            "too many pieces (5 > 4)",
            "1{b[4]f[GTGA]b[4]f[GTGA]b[4]f[GTGA]b[4]f[GTGA]b[4]u[8]}2{r:}",
            "split across",
        ),
        (
            "two variable pieces in one level",
            "1{b[8-9]f[GTGA]b[8-9]f[GACA]b[9]u[8]}2{r:}",
            "variable-length pieces",
        ),
    ];
    for (label, geom, expect_sub) in cases {
        match compile(geom) {
            Ok(_) => {
                all_ok = false;
                println!("  {:<44} FAIL (accepted, should be rejected)", label);
            }
            Err(e) => {
                let ok = e.contains(expect_sub);
                all_ok &= ok;
                println!(
                    "  {:<44} {}  ({})",
                    label,
                    if ok { "PASS" } else { "FAIL" },
                    e.lines().next().unwrap_or("")
                );
            }
        }
    }

    println!(
        "\n==== {} ====",
        if all_ok {
            "ALL CHECKS PASSED"
        } else {
            "SOME CHECKS FAILED"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
}
