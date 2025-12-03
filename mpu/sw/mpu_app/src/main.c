#include <stdio.h>
#include "xil_printf.h"
#include "xaxidma.h"
#include "xparameters.h"
#include "xil_cache.h"

// params
#define DMA_DEV_ID      XPAR_XAXIDMA_0_BASEADDR
#define MEM_BASE_ADDR   0x01000000 // DDR address for our buffer
#define TX_BUFFER_BASE  (MEM_BASE_ADDR + 0x00100000)

// packet size in bytes
#define PACKET_LEN      100

// global variables
XAxiDma DmaInst;

// function prototypes
int init_dma();
int send_packet(u8 *data, int len);

int main()
{
    print("\r\n--- MPU System Test ---\r\n");

    // init dma
    if (init_dma() != XST_SUCCESS) {
        print("DMA Init Failed!\r\n");
        return -1;
    }
    print("DMA Initialized.\r\n");

    // prepare data
    u8 *TxBufferPtr = (u8 *)TX_BUFFER_BASE;
    for(int i = 0; i < PACKET_LEN; i++) {
        TxBufferPtr[i] = i; // ramp pattern: 0, 1, 2, ...
    }

    // flush cache (CRITICAL for DMA)
    // the DMA reads from physical RAM, so we must ensure CPU writes are flushed
    Xil_DCacheFlushRange((UINTPTR)TxBufferPtr, PACKET_LEN);

    // send data
    print("Sending packet via DMA...\r\n");
    int status = send_packet(TxBufferPtr, PACKET_LEN);

    if (status == XST_SUCCESS) {
        print("DMA Transfer Submitted.\r\n");
    } else {
        print("DMA Transfer Failed.\r\n");
    }

    // loop
    while(1) {
        // in a real app, we might wait for a 'Done' interrupt or send more data
        // for now, just spin
    }

    return 0;
}

int init_dma() {
    XAxiDma_Config *CfgPtr;

    // use XAxiDma_LookupConfig with the base address directly if device ID lookup fails
    
    CfgPtr = XAxiDma_LookupConfig(DMA_DEV_ID);
    if (!CfgPtr) {
        print("No config found for DMA\r\n");
        return XST_FAILURE;
    }

    int Status = XAxiDma_CfgInitialize(&DmaInst, CfgPtr);
    if (Status != XST_SUCCESS) {
        print("Initialization failed\r\n");
        return XST_FAILURE;
    }

    // disable interrupts, we use polling mode
    XAxiDma_IntrDisable(&DmaInst, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

    return XST_SUCCESS;
}

int send_packet(u8 *data, int len) {
    int Status;

    // check if DMA is busy
    if (XAxiDma_Busy(&DmaInst, XAXIDMA_DMA_TO_DEVICE)) {
        print("DMA is busy...\r\n");
        return XST_FAILURE;
    }

    // start transfer
    Status = XAxiDma_SimpleTransfer(&DmaInst, (UINTPTR)data, len, XAXIDMA_DMA_TO_DEVICE);
    if (Status != XST_SUCCESS) {
        print("DMA Transfer Error\r\n");
        return XST_FAILURE;
    }

    // wait for completion (polling)
    while (XAxiDma_Busy(&DmaInst, XAXIDMA_DMA_TO_DEVICE)) {
        // wait
    }

    return XST_SUCCESS;
}
