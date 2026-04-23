window.DASHBOARD_DATA = {
  project: {
    title: "Adversarial Robustness in Neural Networks",
    subtitle: "A master’s-style research project with a GitHub-ready interactive dashboard",
    dataset: "Executed locally on the scikit-learn digits benchmark (1,797 samples, 10 classes, 8 x 8 grayscale images)",
    note: "Executed results come from the local handwritten-digit proxy. MNIST and CIFAR-10 are included as an extension protocol, not as completed local runs."
  },
  metrics: {
    baseline: { clean: 96.67, fgsm: 21.11, pgd: 0.28, f1: 0.9665 },
    robust: { clean: 96.39, fgsm: 58.89, pgd: 30.28, f1: 0.9634 }
  },
  figures: [
    {
      id: "loss",
      title: "Training dynamics",
      caption: "Adversarial training converges more slowly but remains stable.",
      src: "./assets/loss_curves.png"
    },
    {
      id: "accuracy",
      title: "Validation accuracy",
      caption: "The defended model preserves strong clean-data performance.",
      src: "./assets/accuracy_curves.png"
    },
    {
      id: "impact",
      title: "Attack impact",
      caption: "The baseline collapses rapidly as attack strength increases.",
      src: "./assets/attack_impact.png"
    },
    {
      id: "confusion",
      title: "Confusion shift under PGD",
      caption: "PGD changes the entire error geometry, not only the confidence score.",
      src: "./assets/confusion_compare.png"
    },
    {
      id: "examples",
      title: "Adversarial examples",
      caption: "Small perturbations are enough to redirect predictions.",
      src: "./assets/adversarial_examples.png"
    }
  ],
  downloads: [
    { label: "Download manuscript (DOCX)", href: "./downloads/adversarial_robustness_masters_style.docx" },
    { label: "Download manuscript (PDF)", href: "./downloads/adversarial_robustness_masters_style.pdf" },
    { label: "Download presentation (PPTX)", href: "./downloads/adversarial_robustness_presentation.pptx" }
  ],
  keyFindings: [
    "A highly accurate model can still be almost useless under PGD.",
    "PGD is the decisive stress test in this project.",
    "Adversarial training preserves clean accuracy and improves robustness substantially.",
    "Robustness should be reported separately from natural accuracy."
  ]
};
