/**
 * SIGIL Transaction Handler
 * Connects form to MetaMask + SIGIL API
 */

const API_URL = 'http://127.0.0.1:8000/sigil';
const SEPOLIA_CHAIN_ID = '0xaa36a7';

// Force switch to Sepolia network
async function ensureSepoliaNetwork() {
    try {
        const chainId = await window.ethereum.request({ method: 'eth_chainId' });
        
        if (chainId !== SEPOLIA_CHAIN_ID) {
            console.log('Switching to Sepolia...');
            showStatus('🔄 Switching to Sepolia testnet...', 'info');
            
            try {
                await window.ethereum.request({
                    method: 'wallet_switchEthereumChain',
                    params: [{ chainId: SEPOLIA_CHAIN_ID }],
                });
                showStatus('✅ Switched to Sepolia!', 'success');
                await new Promise(r => setTimeout(r, 1000));
            } catch (switchError) {
                if (switchError.code === 4902) {
                    await window.ethereum.request({
                        method: 'wallet_addEthereumChain',
                        params: [{
                            chainId: SEPOLIA_CHAIN_ID,
                            chainName: 'Sepolia Test Network',
                            nativeCurrency: {
                                name: 'SepoliaETH',
                                symbol: 'ETH',
                                decimals: 18
                            },
                            rpcUrls: ['https://sepolia.infura.io/v3/'],
                            blockExplorerUrls: ['https://sepolia.etherscan.io']
                        }]
                    });
                    showStatus('✅ Sepolia network added!', 'success');
                    await new Promise(r => setTimeout(r, 1000));
                } else {
                    throw switchError;
                }
            }
        }
    } catch (error) {
        throw new Error('Failed to switch to Sepolia. Please switch manually in MetaMask.');
    }
}

// Detect MetaMask
async function detectMetaMask() {
    if (window.ethereum) {
        return window.ethereum;
    }
    
    return new Promise((resolve, reject) => {
        let attempts = 0;
        const checkMetaMask = setInterval(() => {
            attempts++;
            if (window.ethereum) {
                clearInterval(checkMetaMask);
                resolve(window.ethereum);
            } else if (attempts >= 10) {
                clearInterval(checkMetaMask);
                reject(new Error('MetaMask not installed'));
            }
        }, 300);
    });
}

// Connect wallet
async function connectWallet() {
    try {
        const ethereum = await detectMetaMask();
        const accounts = await ethereum.request({ method: 'eth_requestAccounts' });
        console.log('✅ Connected:', accounts[0]);
        
        // Use ethers v6 syntax
        const provider = new ethers.BrowserProvider(ethereum);
        const signer = await provider.getSigner();
        return signer;
    } catch (error) {
        if (error.message.includes('not installed')) {
            showError('MetaMask not detected! Install from https://metamask.io/');
        } else if (error.code === 4001) {
            showError('Connection rejected by user');
        } else {
            showError(error.message);
        }
        throw error;
    }
}

// Show error message
function showError(message) {
    const errorEl = document.getElementById('sdk-error');
    if (errorEl) {
        errorEl.textContent = `❌ ${message}`;
        errorEl.style.display = 'block';
        errorEl.style.padding = '15px';
        errorEl.style.marginTop = '15px';
        errorEl.style.background = 'rgba(255, 77, 77, 0.1)';
        errorEl.style.border = '1px solid rgba(255, 77, 77, 0.3)';
        errorEl.style.borderRadius = '8px';
        errorEl.style.color = '#ff4d4d';
        
        setTimeout(() => {
            errorEl.style.display = 'none';
        }, 5000);
    }
}

// Show status message
function showStatus(message, type = 'info') {
    const statusEl = document.getElementById('sdk-status');
    if (statusEl) {
        statusEl.textContent = message;
        statusEl.style.display = 'block';
        statusEl.style.padding = '15px';
        statusEl.style.marginTop = '15px';
        statusEl.style.borderRadius = '8px';
        
        if (type === 'success') {
            statusEl.style.background = 'rgba(76, 217, 100, 0.1)';
            statusEl.style.border = '1px solid rgba(76, 217, 100, 0.3)';
            statusEl.style.color = '#4cd964';
        } else if (type === 'error') {
            statusEl.style.background = 'rgba(255, 77, 77, 0.1)';
            statusEl.style.border = '1px solid rgba(255, 77, 77, 0.3)';
            statusEl.style.color = '#ff4d4d';
        } else {
            statusEl.style.background = 'rgba(139, 111, 214, 0.1)';
            statusEl.style.border = '1px solid rgba(139, 111, 214, 0.3)';
            statusEl.style.color = '#8B6FD6';
        }
    }
}

// Show receipt (expand form card to show results)
function showReceipt(data) {
    const statusEl = document.getElementById('sdk-status');
    if (statusEl) {
        const truncate = (addr) => addr.substring(0, 10) + '...' + addr.substring(addr.length - 8);
        
        statusEl.innerHTML = `
            <div style="text-align: left;">
                <h3 style="color: #8B6FD6; margin-bottom: 15px; font-size: 18px;">✅ Transaction Complete</h3>
                <div style="display: flex; flex-direction: column; gap: 10px; font-size: 14px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.6);">SIGIL Norm:</span>
                        <span style="color: #fff; font-weight: 600;">${data.signature_norm.toFixed(2)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.6);">TX Hash:</span>
                        <a href="https://sepolia.etherscan.io/tx/${data.tx_hash}" 
                           target="_blank" 
                           style="color: #8B6FD6; text-decoration: none;">
                            ${truncate(data.tx_hash)}
                        </a>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.6);">From:</span>
                        <span style="color: #fff;">${truncate(data.sender)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.6);">To:</span>
                        <span style="color: #fff;">${truncate(data.receiver)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.6);">Amount:</span>
                        <span style="color: #fff; font-weight: 600;">${data.amount} ETH</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: rgba(255,255,255,0.6);">Status:</span>
                        <span style="color: #4cd964;">🛡️ Quantum-Safe ✅</span>
                    </div>
                </div>
            </div>
        `;
        statusEl.style.display = 'block';
        statusEl.style.background = 'rgba(139, 111, 214, 0.15)';
        statusEl.style.border = '1px solid rgba(139, 111, 214, 0.4)';
    }
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();
    
    const submitBtn = document.getElementById('sdk-submit-btn');
    const recipientInput = document.getElementById('sdk-recipient');
    const amountInput = document.getElementById('sdk-amount');
    const messageInput = document.getElementById('sdk-message');
    
    // Clear previous messages
    document.getElementById('sdk-error').style.display = 'none';
    document.getElementById('sdk-status').style.display = 'none';
    
    // Disable button
    submitBtn.disabled = true;
    submitBtn.textContent = '⏳ Processing...';
    submitBtn.style.opacity = '0.6';
    
    try {
        const to = recipientInput.value;
        const amount = amountInput.value;
        const message = messageInput.value || 'SIGIL: Quantum-safe transaction!';
        
        // Step 1: Ensure Sepolia network
        await ensureSepoliaNetwork();
        
        // Step 2: Connect wallet
        showStatus('🦊 Connecting to MetaMask...', 'info');
        const signer = await connectWallet();
        const from = await signer.getAddress();
        console.log('📍 Wallet:', from);
        
        // Step 3: SIGIL signature generation
        showStatus('🔐 Generating SIGIL signature...', 'info');
        const prepareResponse = await fetch(`${API_URL}/prepare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                sender: from, 
                receiver: to, 
                amount: amount, 
                message: message 
            })
        });
        
        if (!prepareResponse.ok) {
            throw new Error(`API error: ${prepareResponse.status}`);
        }
        
        const prepareData = await prepareResponse.json();
        console.log('📥 SIGIL Response:', prepareData);
        
        showStatus(`✅ SIGIL score: ${prepareData.final_score.toFixed(3)} (${prepareData.verdict})`, 'success');
        await new Promise(r => setTimeout(r, 1000));
        
        // Step 4: Check verdict
        if (prepareData.verdict !== "ACCEPT") {
            throw new Error(`SIGIL rejected signature! Score: ${prepareData.final_score.toFixed(3)}`);
        }
        
        // Step 5: Send transaction
        showStatus('🦊 Confirm transaction in MetaMask...', 'info');
        const tx = await signer.sendTransaction({
            to: to,
            value: ethers.parseEther(amount)
        });
        
        console.log('✅ TX sent:', tx.hash);
        showStatus('⏳ Waiting for blockchain confirmation...', 'info');
        
        await tx.wait();
        console.log('✅ TX confirmed!');
        
        // Step 6: Record transaction
        showStatus('📝 Recording transaction...', 'info');
        const recordResponse = await fetch(`${API_URL}/record`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tx_hash: tx.hash,
                sender: from,
                receiver: to,
                amount: amount,
                message: message,
                sigil_signature: prepareData.sigil_signature,
                signature_norm: prepareData.signature_norm
            })
        });
        
        const recordData = await recordResponse.json();
        console.log('📥 Record Response:', recordData);
        
        // Step 7: Show receipt
        showReceipt(recordData.receipt);
        
        // Clear form
        recipientInput.value = '';
        amountInput.value = '';
        messageInput.value = '';
        
    } catch (error) {
        console.error('❌ Error:', error);
        showError(error.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Sign with SIGIL & Send';
        submitBtn.style.opacity = '1';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('sigil-tx-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
        console.log('✅ SIGIL transaction form initialized');
    }
    
    // Check for MetaMask on load
    detectMetaMask()
        .then(() => {
            console.log('✅ MetaMask detected');
        })
        .catch(() => {
            console.warn('⚠️ MetaMask not detected');
            showError('MetaMask not detected. Install from https://metamask.io/');
        });
});
