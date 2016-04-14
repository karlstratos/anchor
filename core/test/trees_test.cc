// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the code for manipulating constituency trees.

#include "gtest/gtest.h"

#include <limits.h>

#include "../trees.h"

// Test class for checking TreeReader.
class TreeReaderTest : public testing::Test {
protected:
    TreeReader tree_reader_;
};

// Tests that a generic tree string is correctly read.
TEST_F(TreeReaderTest, ReadGenericTreeTest) {
    //          TOP
    //           |
    //           AA
    //         / |  \
    //      BBB C*#!  D
    //       |   |    |
    //      bbb  Q    d
    //           |
    //         *-1-*
    Node *root = tree_reader_.CreateTreeFromTreeString(
	"(TOP(AA   (BBB	bbb)    (C*#! (Q *-1-*  )) (D d)))");

    // nullptr -> TOP -> AA
    EXPECT_EQ("TOP", root->nonterminal_string());
    EXPECT_EQ("", root->terminal_string());
    EXPECT_EQ(1, root->NumChildren());
    EXPECT_EQ(nullptr, root->parent());
    EXPECT_EQ(0, root->span_begin());
    EXPECT_EQ(2, root->span_end());

    // TOP -> AA -> BBB C*#1 D
    Node *child1 = root->Child(0);
    EXPECT_EQ("AA", child1->nonterminal_string());
    EXPECT_EQ("", child1->terminal_string());
    EXPECT_EQ(3, child1->NumChildren());
    EXPECT_EQ(root, child1->parent());
    EXPECT_EQ(0, child1->span_begin());
    EXPECT_EQ(2, child1->span_end());

    // AA -> BBB : bbb
    Node *child11 = child1->Child(0);
    EXPECT_EQ("BBB", child11->nonterminal_string());
    EXPECT_EQ("bbb", child11->terminal_string());
    EXPECT_EQ(0, child11->NumChildren());
    EXPECT_EQ(child1, child11->parent());
    EXPECT_EQ(0, child11->span_begin());
    EXPECT_EQ(0, child11->span_end());

    // AA -> C*#! -> Q
    Node *child12 = child1->Child(1);
    EXPECT_EQ("C*#!", child12->nonterminal_string());
    EXPECT_EQ("", child12->terminal_string());
    EXPECT_EQ(1, child12->NumChildren());
    EXPECT_EQ(child1, child12->parent());
    EXPECT_EQ(1, child12->span_begin());
    EXPECT_EQ(1, child12->span_end());

    // C*#! -> Q : *-1-*
    Node *child121 = child12->Child(0);
    EXPECT_EQ("Q", child121->nonterminal_string());
    EXPECT_EQ("*-1-*", child121->terminal_string());
    EXPECT_EQ(0, child121->NumChildren());
    EXPECT_EQ(child12, child121->parent());
    EXPECT_EQ(1, child121->span_begin());
    EXPECT_EQ(1, child121->span_end());

    // AA -> D : d
    Node *child13 = child1->Child(2);
    EXPECT_EQ("D", child13->nonterminal_string());
    EXPECT_EQ("d", child13->terminal_string());
    EXPECT_EQ(0, child13->NumChildren());
    EXPECT_EQ(child1, child13->parent());
    EXPECT_EQ(2, child13->span_begin());
    EXPECT_EQ(2, child13->span_end());

    root->DeleteSelfAndDescendents();
}

// Tests that a singleton tree string correctly read.
TEST_F(TreeReaderTest, ReadSingletonTreeTest) {
    Node *root = tree_reader_.CreateTreeFromTreeString("(A a)");

    // nullptr -> A : a
    EXPECT_EQ("A", root->nonterminal_string());
    EXPECT_EQ("a", root->terminal_string());
    EXPECT_EQ(0, root->NumChildren());
    EXPECT_EQ(nullptr, root->parent());
    EXPECT_EQ(0, root->span_begin());
    EXPECT_EQ(0, root->span_end());

    root->DeleteSelfAndDescendents();
}

// Tests that a tree string with extra brackets is correctly read.
TEST_F(TreeReaderTest, ReadTreeWithExtraBracketsTest) {
    //           /\
    //          A                    A
    //        / | \                  |
    //         / \          ==       B
    //        B                      |
    //        |                      b
    //        b
    Node *root = tree_reader_.CreateTreeFromTreeString(
	"((A () ((B b) ()) ()) ())");

    // nullptr -> A -> B
    EXPECT_EQ("A", root->nonterminal_string());
    EXPECT_EQ("",root->terminal_string());
    EXPECT_EQ(1, root->NumChildren());
    EXPECT_EQ(nullptr, root->parent());
    EXPECT_EQ(0, root->span_begin());
    EXPECT_EQ(0, root->span_end());

    // A -> B : b
    Node *child1 = root->Child(0);
    EXPECT_EQ("B", child1->nonterminal_string());
    EXPECT_EQ("b", child1->terminal_string());
    EXPECT_EQ(0, child1->NumChildren());
    EXPECT_EQ(root, child1->parent());
    EXPECT_EQ(0, child1->span_begin());
    EXPECT_EQ(0, child1->span_end());

    root->DeleteSelfAndDescendents();
}

// Reject a right-unbalanced tree.
TEST_F(TreeReaderTest, RejectRightUnbalancedTree) {
    EXPECT_DEATH({ tree_reader_.CreateTreeFromTreeString("(A (B b)))"); }, "");
}

// Reject a left-unbalanced tree.
TEST_F(TreeReaderTest, RejectLeftUnbalancedTree) {
    EXPECT_DEATH({ tree_reader_.CreateTreeFromTreeString("(A (B b)"); }, "");
}

// Reject invalid singleton trees.
TEST_F(TreeReaderTest, RejectInvalidSingletonTrees) {
    EXPECT_DEATH({ tree_reader_.CreateTreeFromTreeString("(A)"); }, "");
    EXPECT_DEATH({ tree_reader_.CreateTreeFromTreeString("()"); }, "");
}

// Reject nodes with more than two symbols.
TEST_F(TreeReaderTest, RejectNodesWithMoreThanTwoSymbols) {
    EXPECT_DEATH({ tree_reader_.CreateTreeFromTreeString("(A B C)"); }, "");
}

// Reject trees starting with a closing bracket.
TEST_F(TreeReaderTest, RejecTreeStartingWithClosingBracket) {
    EXPECT_DEATH({ tree_reader_.CreateTreeFromTreeString(")"); }, "");
}

// Test class for checking Node. This assumes the correctness of TreeReader.
class NodeTest : public testing::Test {
protected:
    TreeReader tree_reader_;

    // Example for binarization.
    //                   (right, no hor. Markov.)      (left, 2 hor. Markov.)
    //
    //       TOP              TOP                                      TOP
    //        |                |                                        |
    //        A                A                                        A
    //      //|\\             / \                                      / \
    //    / / | \ \    =>    B   A|B                                 A|F  F
    //   / /  |  \ \         |   / \                                /  \  |
    //  B C   D   E F        b  C  A|B~C                        A|F~E   E f
    //  | |   |   | |           |  /   \                        /   \   |
    //  b c   |   e f           c D     A|B~C~D             A|E~D    D  e
    //       /|\                 / \       / \              / \     / \
    //      G H I               G   D|G   E   F            B   C  D|I  I
    //     /  |  \              |    / \  |   |            |   |  /\   |
    //    g   h   i             g   H   I e   f            b   c G  H  i
    //                              |   |                        |  |
    //                              h   i         ,              g  h
    string binarization_example_ =
	"(TOP (A (B b) (C c) (D (G g) (H h) (I i)) (E e) (F f)))";
    string binarization_example_right_binarized_nomarkov_ =
	"(TOP (A (B b) (A|B (C c) (A|B~C (D (G g) (D|G (H h) (I i))) "
	"(A|B~C~D (E e) (F f))))))";
    string binarization_example_right_binarized_markov2_ =
	"(TOP (A (B b) (A|B (C c) (A|B~C (D (G g) (D|G (H h) (I i))) "
	"(A|C~D (E e) (F f))))))";
    string binarization_example_left_binarized_markov2_ =
	"(TOP (A (A|F (A|F~E (A|E~D (B b) (C c)) (D (D|I (G g) (H h)) (I i))) "
	"(E e)) (F f)))";
    string binarization_example_right_binarized_markov1_ =
	"(TOP (A (B b) (A|B (C c) (A|C (D (G g) (D|G (H h) (I i))) "
	"(A|D (E e) (F f))))))";
    string binarization_example_right_binarized_markov0_ =
	"(TOP (A (B b) (A| (C c) (A| (D (G g) (D| (H h) (I i))) "
	"(A| (E e) (F f))))))";

    // Example for collapsing unary productions.
    //
    //         TOP
    //          |
    //          A                         TOP+A
    //       /  |   \                    /  |  \
    //      B   L     C            B+D+G   L+M   C
    //      |   |    / \             / |    |   / \
    //      D   M   E   F          I  J+K   m  E  F+H
    //      |   |   |   |    =>    |   |       |   |
    //      G   m   e   H          i   k       e   h
    //    /  \          |
    //   I    J         h
    //   |    |
    //   i    K
    //        |
    //        k
    string unary_example_ =
	"(TOP (A (B (D (G (I i) (J (K k))))) (L (M m)) (C (E e) (F (H h)))))";
    string unary_example_collapsed_ =
	"(TOP+A (B+D+G (I i) (J+K k)) (L+M m) (C (E e) (F+H h)))";
};

// Tests that nodes are compared correctly.
TEST_F(NodeTest, ComparisonTest) {
    const string &tree1_string = "(TOP (A (B b) (C (D d) (E e) (F f))))";
    const string &tree2_string = "(TOP(A(B b)(C(D d)(E e)(F f))))";
    const string &tree3_string = "(TOP (A (B b) (C (D d) (E z) (F f))))";
    const string &tree4_string = "(TOP (A (Q b) (C (D d) (E e) (F f))))";
    const string &tree5_string = "(TOP (A (B b) (C (D d) (E e) (F f) (G g))))";
    Node *tree1 = tree_reader_.CreateTreeFromTreeString(tree1_string);
    EXPECT_TRUE(tree1->Compare(tree1_string));
    EXPECT_TRUE(tree1->Compare(tree2_string));
    EXPECT_FALSE(tree1->Compare(tree3_string));
    EXPECT_FALSE(tree1->Compare(tree4_string));
    EXPECT_FALSE(tree1->Compare(tree5_string));
    tree1->DeleteSelfAndDescendents();
}

// Tests right binarization without horizontal markovization.
TEST_F(NodeTest, RightBinarizationWithoutHorizontalMarkovization) {
    Node *root = tree_reader_.CreateTreeFromTreeString(binarization_example_);
    root->Binarize("right", 0, INT_MAX);
    EXPECT_TRUE(root->Compare(binarization_example_right_binarized_nomarkov_));
    root->DeleteSelfAndDescendents();
}

// Tests right binarization with 2nd-order horizontal markovization.
TEST_F(NodeTest, RightBinarizationWith2ndOrderHorizontalMarkovization) {
    Node *root = tree_reader_.CreateTreeFromTreeString(binarization_example_);
    root->Binarize("right", 0, 2);
    EXPECT_TRUE(root->Compare(binarization_example_right_binarized_markov2_));
    root->DeleteSelfAndDescendents();
}

// Tests left binarization with 2nd-order horizontal markovization.
TEST_F(NodeTest, LeftBinarizationWith2ndOrderHorizontalMarkovization) {
    Node *root = tree_reader_.CreateTreeFromTreeString(binarization_example_);
    root->Binarize("left", 0, 2);
    EXPECT_TRUE(root->Compare(binarization_example_left_binarized_markov2_));
    root->DeleteSelfAndDescendents();
}

// Tests right binarization with 1st-order horizontal markovization.
TEST_F(NodeTest, RightBinarizationWith1stOrderHorizontalMarkovization) {
    Node *root = tree_reader_.CreateTreeFromTreeString(binarization_example_);
    root->Binarize("right", 0, 1);
    EXPECT_TRUE(root->Compare(binarization_example_right_binarized_markov1_));
    root->DeleteSelfAndDescendents();
}

// Tests right binarization with 0th-order horizontal markovization.
TEST_F(NodeTest, RightBinarizationWith0thOrderHorizontalMarkovization) {
    Node *root = tree_reader_.CreateTreeFromTreeString(binarization_example_);
    root->Binarize("right", 0, 0);
    EXPECT_TRUE(root->Compare(binarization_example_right_binarized_markov0_));
    root->DeleteSelfAndDescendents();
}

// Tests debinarizing right binarization.
TEST_F(NodeTest, DebinarizeRightBinarization) {
    Node *root = tree_reader_.CreateTreeFromTreeString(binarization_example_);
    const string &original_treestring = root->ToString();
    root->Binarize("right", 0, INT_MAX);
    root->Debinarize();
    const string &recovered_treestring = root->ToString();
    EXPECT_EQ(original_treestring, recovered_treestring);
    root->DeleteSelfAndDescendents();
}

// Tests debinarizing left binarization.
TEST_F(NodeTest, DebinarizeLeftBinarization) {
    Node *root = tree_reader_.CreateTreeFromTreeString(binarization_example_);
    const string &original_treestring = root->ToString();
    root->Binarize("left", 0, INT_MAX);
    root->Debinarize();
    const string &recovered_treestring = root->ToString();
    EXPECT_EQ(original_treestring, recovered_treestring);
    root->DeleteSelfAndDescendents();
}

// Tests debinarizing a flawed CNF derivation.
TEST_F(NodeTest, DebinarizingFlawedCNFDerivation) {
    //      A|Y~Z                  A
    //      / \          =>       / \
    //     B   C                 B   C
    //     |   |                 |   |
    //     b   c                 b   c
    Node *root = tree_reader_.CreateTreeFromTreeString("(A|Y~Z (B b) (C c))");
    root->Debinarize();
    EXPECT_TRUE(root->Compare("(A (B b) (C c))"));
    root->DeleteSelfAndDescendents();
}

// Tests if unary productions are correctly collapsed.
TEST_F(NodeTest, RemoveUnaryProductionsTest) {
    Node *root = tree_reader_.CreateTreeFromTreeString(unary_example_);
    root->CollapseUnaryProductions();
    EXPECT_TRUE(root->Compare(unary_example_collapsed_));
    root->DeleteSelfAndDescendents();
}

// Tests if collaped unary productions are correctly expanded.
TEST_F(NodeTest, ExpandUnaryProductionsTest) {
    Node *root = tree_reader_.CreateTreeFromTreeString(unary_example_);
    const string &original_treestring = root->ToString();
    root->CollapseUnaryProductions();
    root->ExpandUnaryProductions();
    const string &recovered_treestring = root->ToString();
    EXPECT_EQ(original_treestring, recovered_treestring);
    root->DeleteSelfAndDescendents();
}

// Binarize and collapse unary productions in different orders. Check we get
// back the original tree in recovery.
TEST_F(NodeTest, PerformBinarizationAndUnaryCollapseInDifferentOrders) {
    // Collapse unary productions, then binarize.
    Node *root = tree_reader_.CreateTreeFromTreeString(unary_example_);
    const string &original_treestring = root->ToString();
    root->CollapseUnaryProductions();
    root->Binarize("right", 0, INT_MAX);
    root->Debinarize();
    root->ExpandUnaryProductions();
    EXPECT_EQ(original_treestring, root->ToString());
    root->DeleteSelfAndDescendents();

    // Binarize, then collapse unary productions.
    root = tree_reader_.CreateTreeFromTreeString(unary_example_);
    root->Binarize("right", 0, INT_MAX);
    root->CollapseUnaryProductions();
    root->ExpandUnaryProductions();
    root->Debinarize();
    EXPECT_EQ(original_treestring, root->ToString());
    root->DeleteSelfAndDescendents();
}

// Tests if null productions are correctly removed.
TEST_F(NodeTest, RemoveNullProductionsTest) {
    //             A                       A
    //           /  \                      |
    //          B    C                     B
    //         / \    \          =>        |
    //        D    E  -NONE-               E
    //      /  \   |   |                   |
    //  -NONE-  F  e   *1                  e
    //     |    |
    //    *0  -NONE-
    //          |
    //         *-1
    Node *root = tree_reader_.CreateTreeFromTreeString(
	"(A (B (D (-NONE- *0) (F (-NONE- *-1))) (E e)) (C (-NONE- *1)))");
    root->RemoveNullProductions();
    EXPECT_TRUE(root->Compare("(A (B (E e)))"));
    root->DeleteSelfAndDescendents();
}

// Tests if a new root node is correctly added.
TEST_F(NodeTest, AddNewRootNodeTest) {
    //                                   TOP
    //                                    |
    //            A        =>             A
    //           / \                     / \
    //          B   C                   B   C
    //          |   |                   |   |
    //          b   c                   b   c
    Node *root = tree_reader_.CreateTreeFromTreeString("(A (B b) (C c))");
    root->AddRootNode();
    EXPECT_TRUE(root->Compare("(TOP (A (B b) (C c)))"));

    // This should fail since the special root is already present.
    EXPECT_DEATH({ root->AddRootNode(); }, "");
    root->DeleteSelfAndDescendents();
}

// Test class for checking TreeSet. This assumes the correctness of TreeReader.
class TreeSetTest : public testing::Test {
protected:
    virtual void SetUp() {
	TreeReader tree_reader;
	Node *t1 = tree_reader.CreateTreeFromTreeString("(A a)");
	Node *t2 = tree_reader.CreateTreeFromTreeString("(Z(A a) (B b) (C c))");
	trees_.AddTree(t1);
	trees_.AddTree(t2);
    }

    virtual void TearDown() { }

    TreeSet trees_;
};

// Tests if TreeSet is correctly constructed and destroyed without issues.
TEST_F(TreeSetTest, DoNothing) {
    EXPECT_EQ(2, trees_.NumTrees());
    EXPECT_EQ(1, trees_.NumInterminalTypes());
    EXPECT_EQ(3, trees_.NumPreterminalTypes());
    EXPECT_EQ(3, trees_.NumTerminalTypes());
}

// Tests if TerminalSequences is correctly initialized from a tree set.
TEST(TerminalSequencesTest, InitializeWithTreeSet) {
    TreeSet trees;
    TreeReader tree_reader;
    Node *t1 = tree_reader.CreateTreeFromTreeString(
	"(TOP(AA   (BBB	bbb)    (C*#! (Q *-1-*  )) (D d)))");
    Node *t2 = tree_reader.CreateTreeFromTreeString("(A a)");
    trees.AddTree(t1);
    trees.AddTree(t2);

    TerminalSequences sequences(&trees);
    EXPECT_EQ(2, sequences.NumSequences());
    EXPECT_EQ("bbb *-1-* d", sequences.Sequence(0)->ToString());
    EXPECT_EQ("a", sequences.Sequence(1)->ToString());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
