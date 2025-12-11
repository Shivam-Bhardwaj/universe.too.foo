#!/bin/bash

# Cloudflare Tunnel Setup Script for Universe

set -e

echo "=== Universe Cloudflare Tunnel Setup ==="

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "Installing cloudflared..."

    # For Debian/Ubuntu
    if command -v apt-get &> /dev/null; then
        echo "Adding Cloudflare package repository..."
        curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloudflare-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/cloudflare-archive-keyring.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
        sudo apt-get update
        sudo apt-get install -y cloudflared
    else
        echo "Please install cloudflared manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
        exit 1
    fi
fi

echo "cloudflared version: $(cloudflared --version)"

# Login to Cloudflare (opens browser)
echo ""
echo "Step 1: Logging into Cloudflare..."
echo "This will open your browser for authentication."
read -p "Press Enter to continue..."
cloudflared tunnel login

# Create tunnel
echo ""
echo "Step 2: Creating tunnel 'universe-server'..."
cloudflared tunnel create universe-server

# Get tunnel ID
TUNNEL_ID=$(cloudflared tunnel list 2>/dev/null | grep universe-server | awk '{print $1}' | head -1)

if [ -z "$TUNNEL_ID" ]; then
    echo "Error: Failed to create tunnel"
    exit 1
fi

echo "Tunnel ID: $TUNNEL_ID"

# Copy config
echo ""
echo "Step 3: Setting up configuration..."
mkdir -p ~/.cloudflared
cp config.yml ~/.cloudflared/config.yml

echo "Configuration copied to ~/.cloudflared/config.yml"

# Create DNS record
echo ""
echo "Step 4: Creating DNS record for universe.too.foo..."
cloudflared tunnel route dns universe-server universe.too.foo

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the tunnel:"
echo "  cloudflared tunnel run universe-server"
echo ""
echo "To run as a systemd service (recommended):"
echo "  sudo cloudflared service install"
echo "  sudo systemctl enable cloudflared"
echo "  sudo systemctl start cloudflared"
echo ""
echo "Your Universe instance will be available at:"
echo "  https://universe.too.foo"
echo ""
echo "Note: DNS propagation may take a few minutes."
