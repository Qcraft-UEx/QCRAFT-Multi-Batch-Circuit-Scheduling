namespace QWalkVChain5Microsoft {

    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Math;

    operation ExecuteCircuitQSharp() : Result[] {

        use q = Qubit[5];

        // === Inicio del circuito ===

        H(q[3]);

        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);

        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);
        X(q[3]);

        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);

        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);

        // U(pi/2, -pi, -pi)
        Rz(-PI(), q[3]);
        Ry(PI() / 2.0, q[3]);
        Rz(-PI(), q[3]);

        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);

        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);
        X(q[3]);

        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);

        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);

        // Segundo U
        Rz(-PI(), q[3]);
        Ry(PI() / 2.0, q[3]);
        Rz(-PI(), q[3]);

        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);

        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);
        X(q[3]);

        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[2], q[4], q[0]);
        ApplyCCNOTChain([q[3], q[1]], q[4]);
        CCNOT(q[3], q[2], q[1]);

        X(q[1]);
        CNOT(q[3], q[2]);
        X(q[2]);

        // === Medición ===
        let results = [
            M(q[0]),
            M(q[1]),
            M(q[2]),
            M(q[3]),
            M(q[4])
        ];

        ResetAll(q);
        return results;
    }
}
