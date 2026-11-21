import json
from pathlib import Path
import textwrap

root = Path(__file__).parent
source = next(root.glob('*Pytorch.ipynb'))
nb = json.loads(source.read_text(encoding='utf-8'))
cells = nb['cells']

def set_source(i, text):
    cells[i]['source'] = textwrap.dedent(text).strip().splitlines(keepends=True)

def cell(kind, text):
    result = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(text).strip().splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result

extra = {}
def after(i, kind, text):
    extra.setdefault(i, []).append(cell(kind, text))

set_source(2, '''
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(42)
torch.set_num_threads(4)
ARTIFACTS = Path("resultats_pytorch")
ARTIFACTS.mkdir(exist_ok=True)
print("PyTorch version:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
''')
set_source(18, '''
# Visualisation facultative : les exercices ne dépendent pas de Graphviz.
import shutil
try:
    from torchviz import make_dot
except ImportError:
    print("Visualisation facultative ignorée : torchviz n'est pas installé.")
else:
    if shutil.which("dot"):
        dot = make_dot(y, params=dict(model.named_parameters()))
        display(dot)
    else:
        print("Visualisation facultative ignorée : exécutable Graphviz absent.")
''')
set_source(44, '''
r2 = 1 - ((y_hat - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"R2 Score = {r2:.3f}")
''')
set_source(52, '''
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay

print(f"Accuracy = {accuracy_score(y_cls, pred):.4f}")
print(f"Recall = {recall_score(y_cls, pred):.4f}")
print(f"Precision = {precision_score(y_cls, pred):.4f}")
print(f"F1 = {f1_score(y_cls, pred):.4f}")
ConfusionMatrixDisplay.from_predictions(y_cls, pred)
plt.show()
''')
set_source(56, ''.join(cells[56]['source']).replace('path = "classifier_state_dict.pt"','path = ARTIFACTS / "classifier_state_dict.pt"'))
after(73, 'markdown', '''
### Réponses — régression des salaires

- À expertise constante, une année supplémentaire augmente le salaire moyen de **600 DH** selon la règle de génération. L'estimation apprise figure ci-dessous.
- La perte ne devient pas nulle car le bruit aléatoire n'est pas expliqué par les deux variables. Le modèle apprend la tendance moyenne.
- **40 ans est une extrapolation** hors de l'intervalle observé [0, 20]. Le calcul est possible, mais sa fiabilité n'est pas établie : la relation peut changer hors du domaine d'entraînement.
- Davantage de bruit augmente en général l'erreur de prédiction et l'incertitude sur les coefficients.
- Si le modèle prédit en milliers de DH à partir de z=(x−μ)/σ, alors les coefficients d'origine sont **w=1000·w_norm/σ** et **b=1000·b_norm−Σ(w·μ)**.
''')
after(73, 'code', '''
with torch.no_grad():
    coefficients_dh = 1000 * model.weight / ecart_type
    biais_dh = 1000 * model.bias - (coefficients_dh * moyenne).sum()
print(f"Ancienneté : {coefficients_dh[0, 0].item():.2f} DH/an (attendu : 600)")
print(f"Expertise : {coefficients_dh[0, 1].item():.2f} DH/niveau (attendu : 1500)")
print(f"Biais : {biais_dh.item():.2f} DH (attendu : 4000)")
print(f"Écart au salaire théorique de 14200 DH : {salaire_predit - 14200:.2f} DH")
''')
iris_title = ''.join(cells[85]['source']).replace('Exercice 3', 'Exercice 2')
set_source(74, iris_title + '\n\n**1. Préparation des données**')
set_source(85, 'Fin de l’exercice Iris. Les scores portent sur seulement 30 fleurs de test : ils dépendent du partage choisi.')
set_source(86, ''.join(cells[86]['source']).replace('Exercice 2', 'Exercice 3'))
after(76, 'markdown', '''
**Réponses :** les quatre colonnes sont la longueur du sépale, sa largeur, la longueur du pétale et sa largeur, en centimètres, puis standardisées. Les étiquettes sont des indices de classes (0, 1, 2) ; `CrossEntropyLoss` attend donc des entiers `torch.long`. Ajuster le scaler sur le test provoquerait une fuite d'information : le test doit rester indépendant de la préparation et de l'apprentissage.
''')
after(84, 'code', '''
from sklearn.metrics import ConfusionMatrixDisplay

iris_mlp = model
torch.manual_seed(42)
iris_linear = nn.Linear(4, 3)
linear_optimizer = torch.optim.Adam(iris_linear.parameters(), lr=0.01)
linear_losses = []
for epoch in range(200):
    iris_linear.train()
    linear_optimizer.zero_grad()
    linear_loss = criterion(iris_linear(X_train), y_train)
    linear_loss.backward()
    linear_optimizer.step()
    linear_losses.append(linear_loss.item())

plt.figure(figsize=(7, 4))
plt.plot(losses, label="MLP : 4 → 8 → 3")
plt.plot(linear_losses, label="Linéaire : 4 → 3")
plt.xlabel("Époque")
plt.ylabel("Entropie croisée")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, classifier) in zip(axes, [("MLP", iris_mlp), ("Linéaire", iris_linear)]):
    classifier.eval()
    with torch.no_grad():
        predicted = classifier(X_test).argmax(dim=1)
    score = (predicted == y_test).float().mean().item()
    print(f"{name} : accuracy test = {score:.2%}")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, labels=[0, 1, 2],
        display_labels=iris.target_names, ax=ax, colorbar=False)
    ax.set_title(f"{name} — {score:.1%}")
plt.tight_layout()
plt.show()

new_flower = torch.tensor(scaler.transform([[5.1, 3.5, 1.4, 0.2]]), dtype=torch.float32)
with torch.no_grad():
    species = iris_mlp(new_flower).argmax(dim=1).item()
print("Espèce prédite :", iris.target_names[species])
''')
after(84, 'markdown', '''
**Comparaison :** les deux modèles utilisent le même partage, les mêmes données standardisées, Adam à 0,01 et 200 époques. Le MLP comporte 67 paramètres et peut apprendre des frontières non linéaires ; le modèle linéaire comporte 15 paramètres. Les scores affichés permettent de comparer cette exécution, mais un test de 30 fleurs et une seule initialisation ne suffisent pas à conclure à la supériorité générale d'un modèle. La nouvelle fleur est normalisée avec le scaler d'entraînement.
''')
s = ''.join(cells[89]['source'])
s = s.replace('precision = (tp / confusion.sum(dim=0).clamp_min(1)).mean()\nrecall = (tp / confusion.sum(dim=1).clamp_min(1)).mean()\nf1_score = 2 * precision * recall / (precision + recall).clamp_min(1e-12)',
'''precision_par_classe = tp / confusion.sum(dim=0).clamp_min(1)
recall_par_classe = tp / confusion.sum(dim=1).clamp_min(1)
precision = precision_par_classe.mean()
recall = recall_par_classe.mean()
f1_par_classe = 2 * precision_par_classe * recall_par_classe / (precision_par_classe + recall_par_classe).clamp_min(1e-12)
f1_score = f1_par_classe.mean()''')
start = s.index('fig, axes = plt.subplots(1, min(6,')
end = s.index('torch.save(', start)
s = s[:start] + '''if misclassified_images:
    fig, axes = plt.subplots(1, min(6, len(misclassified_images)), figsize=(12, 3), squeeze=False)
    for ax, (image, true_label, predicted_label) in zip(axes.flat, misclassified_images[:6]):
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"Réel: {true_label.item()} / Prédit: {predicted_label.item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("Aucune image mal classée.")

''' + s[end:]
s = s.replace('"mnist_mlp.pt"', 'ARTIFACTS / "mnist_mlp.pt"')
s += '''
with torch.no_grad():
    reloaded_predictions = torch.cat([reloaded_model(batch).argmax(dim=1) for batch, _ in test_loader])
assert torch.equal(reloaded_predictions, all_predictions)
print("Rechargement vérifié : toutes les prédictions de test sont identiques.")
'''
set_source(89, s)
after(93, 'markdown', '''
**Réponses :** la forme `[64, 1, 28, 28]` signifie 64 images par lot, 1 canal de gris, 28 pixels de hauteur et 28 de largeur. Le dernier lot peut être plus petit. `shuffle=True` renouvelle l'ordre d'entraînement pour éviter des lots systématiquement liés à l'ordre des données. Le test n'a pas besoin d'être mélangé.
''')
s = ''.join(cells[98]['source']).replace('out_channels=32', 'out_channels=16').replace('in_channels=32', 'in_channels=16').replace('out_channels=64', 'out_channels=32').replace('64 * 7 * 7', '32 * 7 * 7').replace('nn.Linear(128, 10)', 'nn.Dropout(p=0.5),\n            nn.Linear(128, 10)')
set_source(98, s)
after(99, 'markdown', '''
**Réponse :** le schéma impose 16 puis 32 filtres. Les convolutions de noyau 3 et padding 1 conservent la taille spatiale ; les deux max-poolings la réduisent de 28 à 14 puis à 7. On obtient donc **32 × 7 × 7 = 1568 valeurs** par image avant la couche linéaire. Le code est aligné sur ce schéma. Le Dropout demandé est ajouté avant la dernière couche linéaire : il désactive aléatoirement 50 % des activations à l'entraînement et devient inactif en mode `eval()`.
''')
after(105, 'code', '''
model.eval()
cnn_errors = []
with torch.no_grad():
    for images, labels in test_loader:
        predicted = model(images.to(device)).argmax(dim=1).cpu()
        wrong = predicted != labels
        cnn_errors.extend(zip(images[wrong], labels[wrong], predicted[wrong]))
        if len(cnn_errors) >= 10:
            break
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax in axes.flat:
    ax.axis("off")
for ax, (image, truth, prediction) in zip(axes.flat, cnn_errors[:10]):
    ax.imshow((image[0] * 0.3081 + 0.1307).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Réel : {truth.item()} | Prédit : {prediction.item()}")
plt.tight_layout()
plt.show()
print(f"Images mal classées affichées : {min(10, len(cnn_errors))}")

plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), loss_history, marker="o")
plt.xlabel("Époque")
plt.ylabel("Entropie croisée moyenne")
plt.title("Apprentissage du CNN MNIST")
plt.grid(alpha=0.3)
plt.show()

cnn_path = ARTIFACTS / "mnist_cnn.pth"
torch.save(model.state_dict(), cnn_path)
cnn_reloaded = MNISTClassifier().to(device)
cnn_reloaded.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
cnn_reloaded.eval()
with torch.no_grad():
    for batch, _ in test_loader:
        batch = batch.to(device)
        torch.testing.assert_close(cnn_reloaded(batch), model(batch), rtol=0, atol=0)
print("CNN sauvegardé et rechargé : logits identiques sur tout le test.")
''')
result = [cell('markdown', '''
# Corrigé complet — Bases de PyTorch

Ce notebook conserve le cours et complète les quatre exercices : salaires, Iris, MNIST avec un MLP et MNIST avec un CNN. Exécuter les cellules dans l'ordre. Les jeux MNIST sont téléchargés au premier lancement ; les poids sont enregistrés dans `resultats_pytorch/`. Bibliothèques nécessaires : torch, torchvision, matplotlib et scikit-learn. Torchviz et Graphviz sont facultatifs.
''')]
for i, c in enumerate(cells):
    c['source'] = [line.replace('\x0c\nrac', '\\frac').replace('\x0crac', '\\frac') for line in c['source']]
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None
    c.pop('id', None)
    result.append(c)
    result.extend(extra.get(i, []))
nb['cells'] = result
nb['nbformat_minor'] = 4
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
destination = root / 'Activite_pytorch_corrige.ipynb'
destination.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(destination)
