/*
 * main.c - TinyML thermal profiling logger with DWT latency benchmarking
 * Target : STM32 Nucleo-64 F401RE (STM32F401RET6, Cortex-M4, 16 MHz HSI)
 * Style  : pure bare-metal, CMSIS register access only (NO HAL, NO LL, NO printf)
 *
 * CSV line sent every 500 ms (10 fields):
 *   ts,vin,vout,current,power,amb_temp,hs_temp,pred_temp,infer_us,fault_code
 */

#include "stm32f401xe.h"
#include <stdint.h>

#if __has_include("mlp_model.h")
#include "mlp_model.h"
#elif __has_include("ml_model/mlp_model.h")
#include "ml_model/mlp_model.h"
#endif

#if __has_include("mlp_model_new.h")
#include "mlp_model_new.h"
#elif __has_include("New_models/mlp_model_new.h")
#include "New_models/mlp_model_new.h"
#endif

#ifndef BENCHMARK_MODE
#define BENCHMARK_MODE 0
#endif

/* =========================== MODEL SELECTION ============================== */
/*
 * Single variable to control which TinyML model runs:
 *   MODEL_MLP_ORIGINAL (0) : mlp_model.c     (4 inputs: Ploss, Current, Tamb, Theatsink)
 *   MODEL_MLP_NEW      (1) : mlp_model_new.c (2 inputs: Theatsink, Tamb -> 1s Lookahead)
 *
 * To switch models, change 'active_model' directly below or pass -DACTIVE_MODEL=0 or 1.
 */
#define MODEL_MLP_ORIGINAL 0
#define MODEL_MLP_NEW      1

#ifndef ACTIVE_MODEL
#define ACTIVE_MODEL       MODEL_MLP_ORIGINAL
#endif

/* Active model selection variable */
static volatile uint8_t active_model = ACTIVE_MODEL;


/* =========================== USER CONFIGURATION =========================== */
#define SAMPLE_PERIOD_MS   500U

#define INA219_ADDR        0x40U
#define SHUNT_MOHM         100U

#define BME280_ADDR_A      0x76U
#define BME280_ADDR_B      0x77U

#define ADC_VREF_MV        3300U
#define DIV_R_TOP_OHM      20000U
#define DIV_R_BOT_OHM      10000U
#define ADC_OVERSAMPLE     16U

#define I2C_TIMEOUT_MS     5U
#define SPI_TIMEOUT_MS     5U
#define ADC_TIMEOUT_MS     5U

/* ======================== MOSFET POWER CUTOFF ============================= */
/* STM32 Nucleo-64 Pin: PA8 (Arduino Digital Header D7, CN9 Pin 1)
 * High-Side / Active-High Drive: 1 = Power ON (Normal), 0 = Cutoff (Trip >= 90 °C)
 */
#define MOSFET_GPIO_PORT           GPIOA
#define MOSFET_PIN                 8U
#define MOSFET_ACTIVE_HIGH         1U
#define THERMAL_SHUTDOWN_THRESHOLD 9000  /* Disconnect at >= 90.0 °C (9000 cC) */
#define THERMAL_RECOVERY_THRESHOLD 8000  /* Reconnect when cooled below 80.0 °C (hysteresis) */

/* fault bits */
#define F_INA219           0x01U
#define F_BME280           0x02U
#define F_TC_OPEN          0x04U
#define F_TC_SCG           0x08U
#define F_TC_SCV           0x10U
#define F_MAX_COMM         0x20U
#define F_ADC              0x40U
#define F_THERMAL_TRIP     0x80U

/* ============================= TINYML INFERENCE =========================== */
__attribute__((weak)) void mlp_reset_timebase(void)
{
}

__attribute__((weak)) int32_t cnn_predict_temp_cC(int32_t ploss_cmW, int32_t i_cmA, int32_t amb_cC, int32_t hs_cC)
{
    (void)ploss_cmW;
    (void)i_cmA;
    (void)amb_cC;
    return hs_cC;
}

__attribute__((weak)) int32_t mlp_new_predict_temp_cC(int32_t hs_cC, int32_t amb_cC,
                                                       int32_t ploss_cmW, int32_t i_cmA)
{
    (void)amb_cC;
    (void)ploss_cmW;
    (void)i_cmA;
    return hs_cC;
}

/* ================================ TIMEBASE ================================ */
static inline uint32_t millis(void)
{
    return TIM5->CNT;
}

static void delay_ms(uint32_t ms)
{
    uint32_t t0 = millis();
    while ((millis() - t0) < ms) { }
}

static void timer5_init(void)
{
    RCC->APB1ENR |= RCC_APB1ENR_TIM5EN;
    (void)RCC->APB1ENR;
    TIM5->PSC = 16000U - 1U;        /* 16 MHz / 16000 = 1 kHz */
    TIM5->ARR = 0xFFFFFFFFU;
    TIM5->EGR = TIM_EGR_UG;
    TIM5->CR1 = TIM_CR1_CEN;
}

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

/* ============================ CLOCK AND GPIO ============================== */
static void clock_init(void)
{
    RCC->CR |= RCC_CR_HSION;
    while (!(RCC->CR & RCC_CR_HSIRDY)) { }
    RCC->CFGR &= ~(RCC_CFGR_SW | RCC_CFGR_HPRE | RCC_CFGR_PPRE1 | RCC_CFGR_PPRE2);
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_HSI) { }
}

static void gpio_mode(GPIO_TypeDef *g, uint32_t pin, uint32_t mode)
{
    g->MODER = (g->MODER & ~(3U << (pin * 2U))) | (mode << (pin * 2U));
}

static void gpio_speed_high(GPIO_TypeDef *g, uint32_t pin)
{
    g->OSPEEDR |= (3U << (pin * 2U));
}

static void gpio_af(GPIO_TypeDef *g, uint32_t pin, uint32_t af)
{
    uint32_t idx = pin >> 3;
    uint32_t sh  = (pin & 7U) * 4U;
    g->AFR[idx] = (g->AFR[idx] & ~(0xFU << sh)) | (af << sh);
    gpio_mode(g, pin, 2U);
}

static inline void mosfet_power_on(void)
{
#if MOSFET_ACTIVE_HIGH
    MOSFET_GPIO_PORT->BSRR = (1U << MOSFET_PIN);
#else
    MOSFET_GPIO_PORT->BSRR = (1U << (MOSFET_PIN + 16U));
#endif
}

static inline void mosfet_power_off(void)
{
#if MOSFET_ACTIVE_HIGH
    MOSFET_GPIO_PORT->BSRR = (1U << (MOSFET_PIN + 16U));
#else
    MOSFET_GPIO_PORT->BSRR = (1U << MOSFET_PIN);
#endif
}

static void gpio_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN | RCC_AHB1ENR_GPIOBEN;
    (void)RCC->AHB1ENR;

    gpio_mode(GPIOA, 0U, 3U);

    gpio_af(GPIOA, 2U, 7U);
    gpio_af(GPIOA, 3U, 7U);

    GPIOA->BSRR = (1U << 4);
    gpio_mode(GPIOA, 4U, 1U);
    gpio_speed_high(GPIOA, 4U);

    gpio_af(GPIOA, 5U, 5U);
    gpio_af(GPIOA, 6U, 5U);
    gpio_speed_high(GPIOA, 5U);

    GPIOB->OTYPER |= (1U << 8) | (1U << 9);
    GPIOB->PUPDR   = (GPIOB->PUPDR & ~((3U << 16) | (3U << 18))) | (1U << 16) | (1U << 18);
    gpio_speed_high(GPIOB, 8U);
    gpio_speed_high(GPIOB, 9U);
    gpio_af(GPIOB, 8U, 4U);
    gpio_af(GPIOB, 9U, 4U);

    /* MOSFET power cutoff gate control: PA8 (Arduino D7) */
    gpio_mode(MOSFET_GPIO_PORT, MOSFET_PIN, 1U); /* General purpose output */
    gpio_speed_high(MOSFET_GPIO_PORT, MOSFET_PIN);
#if MOSFET_ACTIVE_HIGH
    MOSFET_GPIO_PORT->BSRR = (1U << MOSFET_PIN); /* Power ON by default */
#else
    MOSFET_GPIO_PORT->BSRR = (1U << (MOSFET_PIN + 16U)); /* Power ON by default */
#endif
}

/* ================================== UART ================================== */
static void uart2_init(void)
{
    RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
    (void)RCC->APB1ENR;
    USART2->BRR = 139U;
    USART2->CR1 = USART_CR1_TE | USART_CR1_RE | USART_CR1_UE;
}

static void uart_putc(char c)
{
    while (!(USART2->SR & USART_SR_TXE)) { }
    USART2->DR = (uint8_t)c;
}

static void uart_write(const char *s, uint32_t len)
{
    while (len--) uart_putc(*s++);
    while (!(USART2->SR & USART_SR_TC)) { }
}

static void uart_puts(const char *s)
{
    uint32_t n = 0;
    while (s[n]) n++;
    uart_write(s, n);
}

/* ===================== Integer -> Text (No Printf) ======================== */
static char *put_uint(char *p, uint32_t v)
{
    char tmp[10];
    uint32_t n = 0;
    do { tmp[n++] = (char)('0' + (v % 10U)); v /= 10U; } while (v);
    while (n) *p++ = tmp[--n];
    return p;
}

static char *put_fixed(char *p, int32_t v, uint32_t dec)
{
    uint32_t u, scale = 1U;
    if (v < 0) { *p++ = '-'; u = (uint32_t)(-v); } else { u = (uint32_t)v; }
    for (uint32_t i = 0; i < dec; i++) scale *= 10U;
    p = put_uint(p, u / scale);
    if (dec) {
        uint32_t frac = u % scale;
        *p++ = '.';
        for (uint32_t d = scale / 10U; d > 0U; d /= 10U) {
            *p++ = (char)('0' + ((frac / d) % 10U));
        }
    }
    return p;
}

/* =================================== I2C1 ================================= */
static void i2c1_init(void)
{
    RCC->APB1ENR |= RCC_APB1ENR_I2C1EN;
    (void)RCC->APB1ENR;

    I2C1->CR1 = I2C_CR1_SWRST;
    I2C1->CR1 = 0U;
    I2C1->CR2   = 16U;
    I2C1->CCR   = 80U;
    I2C1->TRISE = 17U;
    I2C1->CR1   = I2C_CR1_PE;
}

static void i2c_recover_bus(void)
{
    GPIOB->MODER = (GPIOB->MODER & ~((3U << 16) | (3U << 18))) | (1U << 16) | (1U << 18);
    for (uint32_t i = 0; i < 9; i++) {
        GPIOB->BSRR = (1U << (8 + 16));
        delay_ms(1);
        GPIOB->BSRR = (1U << 8);
        delay_ms(1);
        if (GPIOB->IDR & (1U << 9)) break;
    }
    GPIOB->BSRR = (1U << (9 + 16));
    delay_ms(1);
    GPIOB->BSRR = (1U << 8);
    delay_ms(1);
    GPIOB->BSRR = (1U << 9);
    delay_ms(1);
    gpio_af(GPIOB, 8U, 4U);
    gpio_af(GPIOB, 9U, 4U);
}

static int i2c_abort(void)
{
    I2C1->CR1 |= I2C_CR1_STOP;
    i2c_recover_bus();
    i2c1_init();
    return -1;
}

static int i2c_wait_sr1(uint32_t mask)
{
    uint32_t t0 = millis();
    while (!(I2C1->SR1 & mask)) {
        if ((millis() - t0) > I2C_TIMEOUT_MS) return -1;
    }
    return 0;
}

static int i2c_wait_idle(void)
{
    uint32_t t0 = millis();
    while (I2C1->SR2 & I2C_SR2_BUSY) {
        if ((millis() - t0) > I2C_TIMEOUT_MS) return -1;
    }
    return 0;
}

static void i2c_clear_addr(void)
{
    volatile uint32_t tmp = I2C1->SR1;
    tmp = I2C1->SR2;
    (void)tmp;
}

static int i2c_send_addr(uint8_t dev, uint8_t rw)
{
    uint32_t t0 = millis();
    I2C1->DR = (uint8_t)((dev << 1) | rw);
    for (;;) {
        uint32_t sr1 = I2C1->SR1;
        if (sr1 & I2C_SR1_ADDR) return 0;
        if (sr1 & I2C_SR1_AF) { I2C1->SR1 &= ~I2C_SR1_AF; return -1; }
        if ((millis() - t0) > I2C_TIMEOUT_MS) return -1;
    }
}

static int i2c_write(uint8_t dev, uint8_t reg, const uint8_t *data, uint8_t len)
{
    if (i2c_wait_idle()) return i2c_abort();

    I2C1->CR1 |= I2C_CR1_START;
    if (i2c_wait_sr1(I2C_SR1_SB))  return i2c_abort();
    if (i2c_send_addr(dev, 0U))    return i2c_abort();
    i2c_clear_addr();

    if (i2c_wait_sr1(I2C_SR1_TXE)) return i2c_abort();
    I2C1->DR = reg;
    for (uint8_t i = 0; i < len; i++) {
        if (i2c_wait_sr1(I2C_SR1_TXE)) return i2c_abort();
        I2C1->DR = data[i];
    }
    if (i2c_wait_sr1(I2C_SR1_BTF)) return i2c_abort();
    I2C1->CR1 |= I2C_CR1_STOP;
    return 0;
}

static int i2c_read(uint8_t dev, uint8_t reg, uint8_t *buf, uint8_t len)
{
    if (len == 0U) return -1;
    if (i2c_wait_idle()) return i2c_abort();

    I2C1->CR1 &= ~I2C_CR1_POS;
    I2C1->CR1 |= I2C_CR1_ACK;

    I2C1->CR1 |= I2C_CR1_START;
    if (i2c_wait_sr1(I2C_SR1_SB))  return i2c_abort();
    if (i2c_send_addr(dev, 0U))    return i2c_abort();
    i2c_clear_addr();
    if (i2c_wait_sr1(I2C_SR1_TXE)) return i2c_abort();
    I2C1->DR = reg;
    if (i2c_wait_sr1(I2C_SR1_BTF)) return i2c_abort();

    I2C1->CR1 |= I2C_CR1_START;
    if (i2c_wait_sr1(I2C_SR1_SB))  return i2c_abort();
    if (i2c_send_addr(dev, 1U))    return i2c_abort();

    if (len == 1U) {
        I2C1->CR1 &= ~I2C_CR1_ACK;
        i2c_clear_addr();
        I2C1->CR1 |= I2C_CR1_STOP;
        if (i2c_wait_sr1(I2C_SR1_RXNE)) return i2c_abort();
        buf[0] = (uint8_t)I2C1->DR;
    } else if (len == 2U) {
        I2C1->CR1 &= ~I2C_CR1_ACK;
        I2C1->CR1 |= I2C_CR1_POS;
        i2c_clear_addr();
        if (i2c_wait_sr1(I2C_SR1_BTF)) return i2c_abort();
        I2C1->CR1 |= I2C_CR1_STOP;
        buf[0] = (uint8_t)I2C1->DR;
        if (i2c_wait_sr1(I2C_SR1_RXNE)) return i2c_abort();
        buf[1] = (uint8_t)I2C1->DR;
        I2C1->CR1 &= ~I2C_CR1_POS;
    } else {
        i2c_clear_addr();
        while (len > 3U) {
            if (i2c_wait_sr1(I2C_SR1_RXNE)) return i2c_abort();
            *buf++ = (uint8_t)I2C1->DR;
            len--;
        }
        if (i2c_wait_sr1(I2C_SR1_BTF)) return i2c_abort();
        I2C1->CR1 &= ~I2C_CR1_ACK;
        *buf++ = (uint8_t)I2C1->DR;
        if (i2c_wait_sr1(I2C_SR1_BTF)) return i2c_abort();
        I2C1->CR1 |= I2C_CR1_STOP;
        *buf++ = (uint8_t)I2C1->DR;
        if (i2c_wait_sr1(I2C_SR1_RXNE)) return i2c_abort();
        *buf++ = (uint8_t)I2C1->DR;
    }
    return 0;
}

/* ================================== INA219 ================================ */
#define INA_REG_CONFIG     0x00U
#define INA_REG_SHUNT      0x01U
#define INA_REG_BUS        0x02U

static uint8_t g_ina_ok = 0;

static int ina219_init(void)
{
    const uint8_t cfg[2] = { 0x3E, 0x67 };
    if (i2c_write(INA219_ADDR, INA_REG_CONFIG, cfg, 2U)) return -1;
    g_ina_ok = 1;
    return 0;
}

static int ina219_read16(uint8_t reg, uint16_t *v)
{
    uint8_t b[2];
    if (i2c_read(INA219_ADDR, reg, b, 2U)) return -1;
    *v = (uint16_t)(((uint16_t)b[0] << 8) | b[1]);
    return 0;
}

static int ina219_read(int32_t *vbus_mV, int32_t *i_cmA)
{
    uint16_t shunt, bus;

    if (!g_ina_ok && ina219_init()) return -1;
    if (ina219_read16(INA_REG_SHUNT, &shunt) || ina219_read16(INA_REG_BUS, &bus)) {
        g_ina_ok = 0;
        return -1;
    }
    *vbus_mV = (int32_t)(bus >> 3) * 4;
    if (*vbus_mV < 1000) {
        *i_cmA = 0; /* Vin < 1.0 V threshold: clamp current to 0 */
    } else {
        *i_cmA = ((int32_t)(int16_t)shunt * 1000) / (int32_t)SHUNT_MOHM;
    }
    return 0;
}

/* ================================== BME280 ================================ */
#define BME_REG_ID         0xD0U
#define BME_REG_RESET      0xE0U
#define BME_REG_CTRL_MEAS  0xF4U
#define BME_REG_CONFIG     0xF5U
#define BME_REG_TEMP_MSB   0xFAU
#define BME_REG_CALIB_T    0x88U

static uint8_t  g_bme_ok   = 0;
static uint8_t  g_bme_addr = BME280_ADDR_A;
static uint16_t dig_T1;
static int16_t  dig_T2, dig_T3;

static int bme280_init(void)
{
    const uint8_t cand[2] = { BME280_ADDR_A, BME280_ADDR_B };
    uint8_t id = 0U, found = 0U, cal[6], v;

    for (uint8_t i = 0; i < 2U; i++) {
        if (i2c_read(cand[i], BME_REG_ID, &id, 1U) == 0 &&
            (id == 0x60U || id == 0x58U || id == 0x57U || id == 0x56U)) {
            g_bme_addr = cand[i];
            found = 1U;
            break;
        }
    }
    if (!found) return -1;

    v = 0xB6U;
    if (i2c_write(g_bme_addr, BME_REG_RESET, &v, 1U)) return -1;
    delay_ms(10U);

    if (i2c_read(g_bme_addr, BME_REG_CALIB_T, cal, 6U)) return -1;
    dig_T1 = (uint16_t)(cal[0] | (cal[1] << 8));
    dig_T2 = (int16_t) (cal[2] | (cal[3] << 8));
    dig_T3 = (int16_t) (cal[4] | (cal[5] << 8));

    v = 0x00U;
    if (i2c_write(g_bme_addr, BME_REG_CONFIG, &v, 1U)) return -1;

    g_bme_ok = 1U;
    return 0;
}

static int bme280_read_temp(int32_t *t_cC)
{
    uint8_t m, d[3];
    int32_t adc_T, var1, var2, t_fine;

    if (!g_bme_ok && bme280_init()) return -1;

    m = 0x21U;
    if (i2c_write(g_bme_addr, BME_REG_CTRL_MEAS, &m, 1U)) { g_bme_ok = 0; return -1; }
    delay_ms(8U);
    if (i2c_read(g_bme_addr, BME_REG_TEMP_MSB, d, 3U))    { g_bme_ok = 0; return -1; }

    adc_T = ((int32_t)d[0] << 12) | ((int32_t)d[1] << 4) | ((int32_t)d[2] >> 4);
    if (adc_T == 0x80000) return -1;

    var1 = ((((adc_T >> 3) - ((int32_t)dig_T1 << 1))) * (int32_t)dig_T2) >> 11;
    var2 = (((((adc_T >> 4) - (int32_t)dig_T1) * ((adc_T >> 4) - (int32_t)dig_T1)) >> 12) * (int32_t)dig_T3) >> 14;
    t_fine = var1 + var2;
    *t_cC = (t_fine * 5 + 128) >> 8;
    return 0;
}

/* =================================== SPI1 ================================= */
#define CS_LOW()   (GPIOA->BSRR = (1U << (4U + 16U)))
#define CS_HIGH()  (GPIOA->BSRR = (1U << 4U))

static void spi1_init(void)
{
    RCC->APB2ENR |= RCC_APB2ENR_SPI1EN;
    (void)RCC->APB2ENR;

    SPI1->CR1 = 0U;
    /* Prescaler = /32 (4U << 3) -> 16 MHz / 32 = 500 kHz for high noise immunity on jumper wires */
    SPI1->CR1 = SPI_CR1_MSTR | (4U << 3) | SPI_CR1_SSM | SPI_CR1_SSI;
    SPI1->CR1 |= SPI_CR1_SPE;
}

static int spi1_xfer8(uint8_t tx, uint8_t *rx)
{
    uint32_t t0 = millis();
    while (!(SPI1->SR & SPI_SR_TXE)) {
        if ((millis() - t0) > SPI_TIMEOUT_MS) return -1;
    }
    SPI1->DR = tx;
    while (!(SPI1->SR & SPI_SR_RXNE)) {
        if ((millis() - t0) > SPI_TIMEOUT_MS) return -1;
    }
    *rx = (uint8_t)SPI1->DR;
    return 0;
}

static int max31855_read_temp(int32_t *t_cC, uint8_t *fault)
{
    uint8_t rx[4];

    (void)SPI1->DR;
    CS_LOW();
    for (volatile uint32_t i = 0; i < 20U; i++) { }

    for (uint8_t i = 0; i < 4U; i++) {
        if (spi1_xfer8(0x00U, &rx[i])) { CS_HIGH(); return -1; }
    }
    while (SPI1->SR & SPI_SR_BSY) { }
    CS_HIGH();

    uint32_t raw = ((uint32_t)rx[0] << 24) | ((uint32_t)rx[1] << 16) | ((uint32_t)rx[2] << 8) | rx[3];

    if (raw == 0xFFFFFFFFU || raw == 0x00000000U) return -1;

    if (raw & (1UL << 16)) {
        uint8_t f = 0U;
        if (raw & (1UL << 0)) f |= F_TC_OPEN;
        if (raw & (1UL << 1)) f |= F_TC_SCG;
        if (raw & (1UL << 2)) f |= F_TC_SCV;
        *fault = f;
        return 1;
    }

    *t_cC = (((int32_t)raw) >> 18) * 25;
    return 0;
}

/* =================================== ADC1 ================================= */
static void adc1_init(void)
{
    RCC->APB2ENR |= RCC_APB2ENR_ADC1EN;
    (void)RCC->APB2ENR;

    ADC1->CR2   = 0U;
    ADC1->SMPR2 = (6U << 0);
    ADC1->SQR1  = 0U;
    ADC1->SQR3  = 0U;
    ADC1->CR2   = ADC_CR2_ADON;
    delay_ms(2U);
}

static int adc1_read_vout_mV(int32_t *vout_mV)
{
    uint32_t sum = 0U, t0;

    for (uint32_t i = 0; i < ADC_OVERSAMPLE; i++) {
        ADC1->CR2 |= ADC_CR2_SWSTART;
        t0 = millis();
        while (!(ADC1->SR & ADC_SR_EOC)) {
            if ((millis() - t0) > ADC_TIMEOUT_MS) return -1;
        }
        sum += ADC1->DR;
    }
    uint32_t raw = sum / ADC_OVERSAMPLE;
    uint32_t pin_mV = (raw * ADC_VREF_MV) / 4095U;
    *vout_mV = (int32_t)((pin_mV * (DIV_R_TOP_OHM + DIV_R_BOT_OHM)) / DIV_R_BOT_OHM);
    return 0;
}

/* ==================================== MAIN ================================ */
int main(void)
{
#if (ACTIVE_MODEL == MODEL_MLP_NEW || (defined(BENCHMARK_MODE) && (BENCHMARK_MODE == 0 || BENCHMARK_MODE == 3)))
    /* Hardware FPU enabled: Grant CP10 & CP11 Full Access */
    SCB->CPACR |= ((3UL << 20) | (3UL << 22));
#else
    /* Hardware FPU physically disabled in silicon: Clear CP10 & CP11 coprocessor access */
    SCB->CPACR &= ~((3UL << 20) | (3UL << 22));
#endif
    __DSB();
    __ISB();

    /* Runtime guarantee: if active_model is set to MODEL_MLP_NEW, enable hardware FPU */
    if (active_model == MODEL_MLP_NEW) {
        SCB->CPACR |= ((3UL << 20) | (3UL << 22));
        __DSB();
        __ISB();
    }

    clock_init();
    gpio_init();
    timer5_init();
    dwt_init();
    uart2_init();
    i2c1_init();
    spi1_init();
    adc1_init();

    uart_puts("# Project ThermoGuard (STM32F401RE Bare-Metal TinyML)\r\n");
    if (active_model == MODEL_MLP_NEW) {
        uart_puts("# Active Model: mlp_model_new (Thermal Lookahead Predictor)\r\n");
    } else {
        uart_puts("# Active Model: mlp_model (Virtual Sensor)\r\n");
    }
#if (BENCHMARK_MODE == 0)
    uart_puts("# Active: Mode 0 (32-bit Float + Hardware FPU)\r\n");
#elif (BENCHMARK_MODE == 1)
    uart_puts("# Active: Mode 1 (32-bit Int Without FPU - Normal Division)\r\n");
#elif (BENCHMARK_MODE == 2)
    uart_puts("# Active: Mode 2 (32-bit Int Without FPU - PoT Bit-Shift)\r\n");
#elif (BENCHMARK_MODE == 3)
    uart_puts("# Active: Mode 3 (16-bit Float + Hardware FPU)\r\n");
#elif (BENCHMARK_MODE == 4)
    uart_puts("# Active: Mode 4 (16-bit Int Without FPU - Normal Division)\r\n");
#elif (BENCHMARK_MODE == 5)
    uart_puts("# Active: Mode 5 (16-bit Int Without FPU - PoT + ARM SMLAD SIMD)\r\n");
#endif
    if (ina219_init())  uart_puts("# WARN: INA219 not responding\r\n");
    if (bme280_init())  uart_puts("# WARN: BME280/BMP280 not responding\r\n");

    uint32_t next = millis();

    for (;;) {
        if ((int32_t)(millis() - next) < 0) continue;
        next += SAMPLE_PERIOD_MS;

        uint32_t ts = millis();
        uint8_t  fault = 0U;
        int32_t  vin_mV = 0, vout_mV = 0, i_cmA = 0, amb_cC = 0, hs_cC = 0;
        int32_t  pred_cC = 0, ploss_cmW = 0;
        uint32_t infer_us = 0;

        /* INA219: 12 V input rail */
        if (ina219_read(&vin_mV, &i_cmA)) {
            vin_mV = 0; i_cmA = 0;
            fault |= F_INA219;
        }

        /* ADC: regulator 8 V output */
        if (adc1_read_vout_mV(&vout_mV)) {
            vout_mV = 0;
            fault |= F_ADC;
        }

        /* BME280: ambient */
        if (bme280_read_temp(&amb_cC)) {
            amb_cC = 0;
            fault |= F_BME280;
        }

        /* MAX31855: heatsink / regulator temperature */
        {
            uint8_t tf = 0U;
            int r = max31855_read_temp(&hs_cC, &tf);
            if (r == 1)       { hs_cC = 0; fault |= tf; }
            else if (r < 0)   { hs_cC = 0; fault |= F_MAX_COMM; }
        }

        /* Anti-glitch freeze filter for measured heatsink temperature:
         * If temperature drops to 0 (SPI noise) or fluctuates >= 10.0 °C (1000 cC)
         * for only 1 or 2 points, freeze display to the last valid stable temperature.
         * Only accept the change if it persists for >= 3 consecutive samples. */
        static int32_t s_clean_hs_cC = 0;
        static uint8_t s_hs_glitch_cnt = 0;

        if (s_clean_hs_cC == 0) {
            if (hs_cC > 0) {
                s_clean_hs_cC = hs_cC;
            }
        } else {
            int32_t diff_hs = hs_cC - s_clean_hs_cC;
            if (diff_hs < 0) diff_hs = -diff_hs;

            if (hs_cC <= 0 || diff_hs >= 1000) {
                s_hs_glitch_cnt++;
                if (s_hs_glitch_cnt <= 2) {
                    hs_cC = s_clean_hs_cC; /* Freeze to last stable reading */
                } else {
                    /* Persisted for > 2 samples: accept legitimate step change */
                    s_clean_hs_cC = hs_cC;
                    s_hs_glitch_cnt = 0;
                }
            } else {
                s_clean_hs_cC = hs_cC;
                s_hs_glitch_cnt = 0;
            }
        }

        /* Automated thermal cutoff protection:
         * Disconnects power source when heatsink temperature reaches >= 90.0 °C.
         * Reconnects when cooled down below 80.0 °C (hysteresis). */
        static uint8_t s_thermal_tripped = 0U;

        if (hs_cC >= THERMAL_SHUTDOWN_THRESHOLD) {
            if (!s_thermal_tripped) {
                s_thermal_tripped = 1U;
                mosfet_power_off();
            }
        } else if (s_thermal_tripped && (hs_cC < THERMAL_RECOVERY_THRESHOLD)) {
            s_thermal_tripped = 0U;
            mosfet_power_on();
        }

        if (s_thermal_tripped) {
            fault |= F_THERMAL_TRIP;
        }

        /* Persistent thermal prediction state (for smooth passive cooling decay) */
        static int32_t s_pred_cC = 0;

        /* Threshold Rule: If Vin < 8500 mV (8.5 V), regulator is unpowered -> COOLING PHASE
         * (KA7808 requires Vin >= 9.5 V to regulate 8.0 V; avoids 30s cap discharge noise) */
        if (vin_mV < 8500) {
            i_cmA = 0;
            ploss_cmW = 0;
            mlp_reset_timebase();

            /* Newton's Law of Passive Cooling:
             * Tau = 208 s (measured from heatsink dataset), dt = 0.5 s
             * Alpha_Q15 = round(exp(-0.5 / 208.0) * 32768) = 32689 */
            if (s_pred_cC == 0) {
                s_pred_cC = (hs_cC > amb_cC) ? hs_cC : amb_cC; /* Initialize to actual heatsink temp if hot at boot */
            }
            int32_t delta = s_pred_cC - amb_cC;
            if (delta > 0) {
                delta = (int32_t)(((int64_t)delta * 32689LL) >> 15);
                s_pred_cC = amb_cC + delta;
            } else {
                s_pred_cC = amb_cC;
            }
            pred_cC = s_pred_cC;
            infer_us = 1U;
        } else {
            if (!(fault & (F_INA219 | F_ADC))) {
                /* Regulator dissipation: (Vin - Vout) * Iin */
                int64_t dv = (int64_t)vin_mV - (int64_t)vout_mV;
                if (dv < 0) dv = 0;
                ploss_cmW = (int32_t)((dv * (int64_t)i_cmA) / 1000);
            } else {
                ploss_cmW = 0;
            }

            /* Benchmark TinyML inference latency via DWT */
            uint32_t t_start = DWT->CYCCNT;
            if (active_model == MODEL_MLP_NEW) {
                /* 4-input stateless model: heatsink, ambient, power, current.
                 * No elapsed_time counter -- starting mid-heating is fully supported. */
                pred_cC = mlp_new_predict_temp_cC(hs_cC, amb_cC, ploss_cmW, i_cmA);
            } else {
                pred_cC = cnn_predict_temp_cC(ploss_cmW, i_cmA, amb_cC, hs_cC);
            }
            uint32_t infer_cycles = DWT->CYCCNT - t_start;
            infer_us = infer_cycles / 16U; /* 16 MHz HSI: 16 cycles = 1 us */

            /* Physical Invariant:
             * While powered, the regulator cannot be colder than the currently measured heatsink.
             * Prevents massive tracking errors when starting mid-heating. */
            int32_t lower_bound = (hs_cC > amb_cC) ? hs_cC : amb_cC;
            if (pred_cC < lower_bound) {
                pred_cC = lower_bound;
            }
            if (pred_cC > 12500) {
                pred_cC = 12500;
            }
        }

        /* Anti-glitch freeze filter for predicted temperature:
         * Suppress transient prediction fluctuations >= 10.0 °C (1000 cC) lasting <= 2 samples */
        static int32_t s_clean_pred_cC = 0;
        static uint8_t s_pred_glitch_cnt = 0;

        if (s_clean_pred_cC == 0) {
            if (pred_cC > 0) {
                s_clean_pred_cC = pred_cC;
            }
        } else {
            int32_t diff_pred = pred_cC - s_clean_pred_cC;
            if (diff_pred < 0) diff_pred = -diff_pred;

            if (diff_pred >= 1000) {
                s_pred_glitch_cnt++;
                if (s_pred_glitch_cnt <= 2) {
                    pred_cC = s_clean_pred_cC; /* Freeze display to last stable reading */
                } else {
                    /* Persisted for > 2 samples: accept legitimate step change */
                    s_clean_pred_cC = pred_cC;
                    s_pred_glitch_cnt = 0;
                }
            } else {
                s_clean_pred_cC = pred_cC;
                s_pred_glitch_cnt = 0;
            }
        }

        s_pred_cC = pred_cC;

        /* Build and send the CSV line (10 fields) */
        char  line[144];
        char *p = line;
        p = put_uint (p, ts);            *p++ = ',';
        p = put_fixed(p, vin_mV, 3U);    *p++ = ',';
        p = put_fixed(p, vout_mV, 3U);   *p++ = ',';
        p = put_fixed(p, i_cmA, 2U);     *p++ = ',';
        p = put_fixed(p, ploss_cmW, 2U); *p++ = ',';
        p = put_fixed(p, amb_cC, 2U);    *p++ = ',';
        p = put_fixed(p, hs_cC, 2U);     *p++ = ',';
        p = put_fixed(p, pred_cC, 2U);   *p++ = ',';
        p = put_uint (p, infer_us);      *p++ = ',';
        p = put_uint (p, fault);
        *p++ = '\r'; *p++ = '\n';
        uart_write(line, (uint32_t)(p - line));
    }
}
