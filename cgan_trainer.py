import torch
import torch.nn as nn
import torch.optim as optim
import time

# ==========================================
# 1. THE ADVANCED GENERATOR
# ==========================================
class AdvancedGenerator(nn.Module):
    def __init__(self):
        super(AdvancedGenerator, self).__init__()

        # Encoder: Extracting spatial geometry from the 256x256 image
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 1024) # Latent environment vector
        )

        # Decoder: Transforming the latent vector into a continuous 1D waveform
        self.decoder = nn.Sequential(
            # Unflatten to (Batch, Channels, Length) -> (Batch, 1, 1024)
            nn.Unflatten(1, (1, 1024)),

            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.InstanceNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 2, kernel_size=7, padding=3) # Outputs exact (Batch, 2, 1024)
        )

    def forward(self, img):
        latent_features = self.encoder(img)
        rf_waveform = self.decoder(latent_features)
        return rf_waveform

# ==========================================
# 2. THE LSGAN DISCRIMINATOR
# ==========================================
class LSGANDiscriminator(nn.Module):
    def __init__(self):
        super(LSGANDiscriminator, self).__init__()

        self.img_processor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 128 * 128, 512)
        )

        self.rf_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 1024, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.judge = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1) # LSGAN Output: No Sigmoid
        )

    def forward(self, img, rf):
        img_features = self.img_processor(img)
        rf_features = self.rf_processor(rf)
        combined = torch.cat((img_features, rf_features), dim=1)
        return self.judge(combined)

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
def train_cgan(dataloader, num_epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Initiating LSGAN Training on {device.type.upper()} ---")

    generator = AdvancedGenerator().to(device)
    discriminator = LSGANDiscriminator().to(device)

    # Hybrid Loss Setup
    criterion_GAN = nn.MSELoss()
    criterion_Pixel = nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        start_time = time.time()
        g_loss_avg = 0.0
        d_loss_avg = 0.0

        for i, (imgs, real_rfs) in enumerate(dataloader):
            imgs, real_rfs = imgs.to(device), real_rfs.to(device)
            batch_size = imgs.size(0)

            valid_labels = torch.ones((batch_size, 1), device=device, dtype=torch.float32)
            fake_labels = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            real_loss = criterion_GAN(discriminator(imgs, real_rfs), valid_labels)
            fake_rfs = generator(imgs)
            fake_loss = criterion_GAN(discriminator(imgs, fake_rfs.detach()), fake_labels)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            g_loss_gan = criterion_GAN(discriminator(imgs, fake_rfs), valid_labels)
            g_loss_pixel = criterion_Pixel(fake_rfs, real_rfs)

            # Combine losses
            g_total_loss = g_loss_gan + (g_loss_pixel * 10.0)

            g_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()

            d_loss_avg += d_loss.item()
            g_loss_avg += g_total_loss.item()

        d_loss_avg /= len(dataloader)
        g_loss_avg /= len(dataloader)
        elapsed = time.time() - start_time

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:03d}/{num_epochs}] | D loss: {d_loss_avg:.4f} | G loss: {g_loss_avg:.4f} | Time: {elapsed:.2f}s")

    save_path = "/content/drive/MyDrive/dataset/processed_data/trained_generator_v2.pth"
    torch.save(generator.state_dict(), save_path)
    print(f"\nTraining Complete! Model securely saved to: {save_path}")

# Note: We assume 'test_loader' is still in memory from the previous step!
train_cgan(test_loader, num_epochs=500)
