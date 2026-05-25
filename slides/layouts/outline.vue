<script setup lang="ts">
import { useSlideContext, useNav } from '@slidev/client'

const { $slidev } = useSlideContext()
const { currentPage, total } = useNav()
const title = $slidev?.configs?.title ?? ''
const speaker = $slidev?.configs?.author ?? ''
</script>

<template>
  <div class="slidev-layout epfl-outline" style="padding: 56px 60px 52px 60px;">
    <div class="epfl-chrome-rule" />
    <img src="/epfl/logo.svg" class="epfl-chrome-logo" alt="EPFL" />

    <div class="outline-eyebrow">Outline</div>

    <div class="outline-body">
      <slot />
    </div>

    <div class="epfl-chrome-footer">
      <div class="meta">
        <span v-if="title">{{ title }}</span>
        <span v-if="speaker" class="sep">·</span>
        <span v-if="speaker">{{ speaker }}</span>
      </div>
      <div>{{ currentPage }} / {{ total }}</div>
    </div>
  </div>
</template>

<style scoped>
.outline-eyebrow {
  font-size: 0.65rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--epfl-text-soft);
  margin: 8px 0 28px 0;
}
.outline-body :deep(ol),
.outline-body :deep(ul) {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0 56px;
  counter-reset: outline;
}
.outline-body :deep(li) {
  counter-increment: outline;
  padding: 14px 0 14px 48px;
  border-top: 1px solid var(--epfl-rule);
  position: relative;
  font-size: 1.0rem;
  color: var(--epfl-text);
}
.outline-body :deep(li::before) {
  content: counter(outline, decimal-leading-zero);
  position: absolute;
  left: 0;
  top: 50%;
  transform: translateY(-50%);
  color: var(--epfl-text-soft);
  font-weight: 400;
  font-size: 0.8rem;
  letter-spacing: 0.05em;
}
</style>
